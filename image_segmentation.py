import cv2
import numpy as np
from ultralytics import YOLO

img_path = "inference/test.jpg"  # Resim dosyanızın yolunu buraya ekleyin
model_path = "C:/Users/aytur/Desktop/3_image_segmentation/runs/segment/yolov8_car_part_segmentation/weights/best.pt"  # Model dosyanızın yolunu buraya ekleyin
font = cv2.FONT_HERSHEY_SIMPLEX

# Resmi yükle
img = cv2.imread(img_path)

# YOLO modelini yükle
model = YOLO(model_path)

# Verileri çıkaran fonksiyon
def extract_data(img, model):
    h, w, ch = img.shape
    results = model.predict(source=img.copy(), save=False, save_txt=False)
    result = results[0]
    seg_contour_idx = []

    # Segmentasyon verilerini normalize et ve uygun formata getir
    for seg in result.masks.xyn:
        seg[:, 0] = seg[:, 0] * w
        seg[:, 1] = seg[:, 1] * h
        segment = np.array(seg, dtype=np.int32)
        seg_contour_idx.append(segment)

    # Bounding box'ları ve diğer bilgileri al
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float")
    class_names = result.names

    return bboxes, class_ids, seg_contour_idx, scores, class_names

# Verileri çıkar
bboxes, class_ids, seg_contour_idx, scores, class_names = extract_data(img, model)

# Bounding box'ları ve segmentasyonları çiz
for box, class_id, segmentation_id, score in zip(bboxes, class_ids, seg_contour_idx, scores):
    (xmin, ymin, xmax, ymax) = box

    # Segmentasyonu doldur
    cv2.fillPoly(img, pts=[segmentation_id], color=(0, 0, 255))

    # Sınıf ismi ve doğruluk skorunu ekle
    class_name = class_names[class_id.item()]
    score = score * 100
    text = f"{class_name}: %{score:.2f}"
    
    # Bounding box ve etiket ekle
    cv2.putText(img, text, (xmin, ymin - 10), font, 0.5, (0, 255, 0), 2)

# İşlenmiş resmi ekranda göster
cv2.imshow("Processed Image", img)

# Ekranda resmi kapatmak için bir tuşa basmayı bekle
cv2.waitKey(0)
cv2.destroyAllWindows()
