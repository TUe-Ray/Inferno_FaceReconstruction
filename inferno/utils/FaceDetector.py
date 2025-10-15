from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from face_alignment.utils import flip, get_preds_fromhm
import mediapipe as mp
# from memory_profiler import profile


def save_landmark(fname, landmark, landmark_type):
    with open(fname, "wb") as f:
        pkl.dump(landmark_type, f)
        pkl.dump(landmark, f)


def load_landmark(fname):
    with open(fname, "rb") as f:
        landmark_type = pkl.load(f)
        landmark = pkl.load(f)
    return landmark_type, landmark


def save_landmark_v2(fname, landmark, landmark_confidence, landmark_type):
    with open(fname, "wb") as f:
        pkl.dump(landmark_type, f)
        pkl.dump(landmark_confidence, f)
        pkl.dump(landmark, f)


def load_landmark_v2(fname):
    with open(fname, "rb") as f:
        landmark_type = pkl.load(f)
        landmark_confidence = pkl.load(f)
        landmark = pkl.load(f)
    return landmark_type, landmark_confidence, landmark


class FaceDetector(ABC):

    @abstractmethod
    def run(self, image, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)


    def landmarks_from_batch_no_face_detection(self, images): 
        """
        This function is used to get the landmarks from a batch of images without face detection. 
        Input: 
            images: a batch of images, shape (N, C, H, W), image range [0, 1]
        Returns:
            landmarks: a list of landmarks, each landmark is a numpy array of shape (N, 68, 2), the position is relative ([0, 1])
            landmark_scores: a list of landmark scores, each landmark score is a numpy array of shape (N, 1) or None if no score is available
        """
        raise NotImplementedError()

    def optimal_landmark_detector_im_size(self): 
        """
        This function returns the optimal image size for the landmark detector. 
        Returns:
            optimal_im_size: int
        """
        raise NotImplementedError()

    def landmark_type(self): 
        """
        This function returns the type of landmarks. 
        Returns:
            landmark_type: str
        """
        raise NotImplementedError()

class MediaPipeMeshFD:
    """
    用 MediaPipe Face Mesh 取得臉部 468/478 個 landmark，
    以 landmark 的外接矩形當作 bbox，回傳與 FAN().run(image) 相容的介面：
      - bboxes: List[[left, top, right, bottom], ...]（像素座標）
      - bbox_type: 'bbox'
    """
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=True,       # 設 True 則 478 點（含虹膜）
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_mesh = mp.solutions.face_mesh

    def _to_uint8(self, image_rgb):
        # mediapipe 預期 uint8 RGB
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                return (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
            return np.clip(image_rgb, 0, 255).astype(np.uint8)
        return image_rgb

    def run(self, image_rgb):
        """
        參數:
            image_rgb: (H, W, 3) 的 RGB 影像
        回傳:
            bboxes: List of [l, t, r, b]（int, 像素座標）
            bbox_type: 'bbox'
        """
        img = self._to_uint8(image_rgb)
        H, W = img.shape[:2]
        bboxes = []

        # 每次呼叫時開關 context，避免長時間佔用資源
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as face_mesh:

            res = face_mesh.process(img)
            if not res.multi_face_landmarks:
                return [], 'bbox'

            for face_lms in res.multi_face_landmarks:
                xs = [lm.x for lm in face_lms.landmark]
                ys = [lm.y for lm in face_lms.landmark]
                # 相對座標 → 像素
                x_min = max(0, int(min(xs) * W))
                x_max = min(W - 1, int(max(xs) * W))
                y_min = max(0, int(min(ys) * H))
                y_max = min(H - 1, int(max(ys) * H))

                # 保障至少 1px 大小
                if x_max <= x_min: x_max = min(W - 1, x_min + 1)
                if y_max <= y_min: y_max = min(H - 1, y_min + 1)

                bboxes.append([x_min, y_min, x_max, y_max])

            # 沒有 score，按框面積由大到小排序（讓 bbox[0] 通常是主臉）
            bboxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            return bboxes, 'bbox'


class FAN(FaceDetector):

    def __init__(self, device = 'cuda', threshold=0.5, mode='2D'):
        import face_alignment
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.flip_input = False
        if mode == '2D':
            try:
                mode = face_alignment.LandmarksType._2D
            except AttributeError:
                mode = face_alignment.LandmarksType.TWO_D
        elif mode == '2.5D':
            try:
                mode = face_alignment.LandmarksType._2halfD
            except AttributeError:
                mode = face_alignment.LandmarksType.TWO_HALF_D
        elif mode == '3D':
            try:
                mode = face_alignment.LandmarksType._3D
            except AttributeError:
                mode = face_alignment.LandmarksType.THREE_D
        else:
            raise ValueError('mode must be 2D or 3D')
        self.model = face_alignment.FaceAlignment(mode,
                                                  device=str(device),
                                                  flip_input=self.flip_input,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    # @profile
    def run(self, image, with_landmarks=True, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'

    @torch.no_grad()
    def landmarks_from_batch_no_face_detection(self, images):
        out = self.model.face_alignment_net(images).detach()
        if self.flip_input:
            out += flip(self.model.face_alignment_net(flip(images)).detach(), is_label=True)

        out = out.cpu().numpy()
        center = None 
        scale = None
        B = out.shape[0]
        pts, pts_img, scores = get_preds_fromhm(out, center, scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(B, 68, 2) * 4, pts_img.view(B, 68, 2)
        scores = scores
        pts /= images.shape[-1]
        pts = pts.cpu().numpy()
        pts_img = pts_img.cpu().numpy()
        return pts, scores
    
    def optimal_landmark_detector_im_size(self): 
        # this number is taken from the crop size used in the face_alignment library
        # function in face_alignment.utils.crop : def crop(image, center, scale, resolution=256.0):
        return 256 

    def landmark_type(self): 
        return 'kpt68'


class MTCNN(FaceDetector):

    def __init__(self, device = 'cuda'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True, device=device)

    def run(self, input, **kwargs):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [], 'bbox'
        else:
            bboxes = []
            for i in range(out.shape[0]):
                bbox = out[0][0].squeeze()
                bboxes += [bbox]
            return bboxes, 'bbox'

    def landmark_type(self): 
        return 'bbox'