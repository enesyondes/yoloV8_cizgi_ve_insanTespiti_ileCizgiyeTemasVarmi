import cv2
import numpy as np
from ultralytics import YOLO

class TruckProtection:
    def __init__(self):
        self.cameraID = "enes2.mp4"         # video'nun dosya yolunu ver
        self.video = cv2.VideoCapture(self.cameraID)        # parantez içini 0 yaparak canlı kameradan görüntü alabiliriz
        
        if self.cameraID != 0 or self.cameraID != 1:
            self.cameraID = "Source is mp4 file"
            self.cameraName = self.video.get(cv2.CAP_PROP_BACKEND)
        self.cameraWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cameraHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cameraFPS = self.video.get(cv2.CAP_PROP_FPS)
        
        if not self.video.isOpened():
            print("Kamera Açılamadı. Lütfen Kamera ID'sini Doğru Girdiğinizden Emin Olun!")
        else:
            print(f"Kamera Bilgileri:\nKamera ID: {self.cameraID}\nKamera İsmi: {self.cameraName}\nKamera Çözünürlüğü: ({self.cameraWidth}, {self.cameraHeight})\nFPS: {int(self.cameraFPS)}")
        
        self.model = YOLO("yolov8n.pt")
        #self.model.export(format="openvino")
        #self.model = YOLO("yolov8n_openvino_model/")
        
        self.warningStatus = 0 


    def drawTrapezoidal(self, image):
        img = image

        # çizgilerin noktaları
        self.pts_red = np.array([[324, 247], [411, 246], [181, 478], [550, 475]], np.int32) #sol üst, sağ üst, sol alt, sağ alt
        self.pts_yellow = np.array([[246, 247], [484, 246], [0, 386], [837, 475]], np.int32) #sol üst, sağ üst, sol alt, sağ alt
        
        # kırmızı çizgiler
        cv2.line(img=img, pt1=self.pts_red[0], pt2=self.pts_red[2], color=(0, 0, 255), thickness=2)
        cv2.line(img=img, pt1=self.pts_red[1], pt2=self.pts_red[3], color=(0, 0, 255), thickness=2)

        # sarı çizgiler
        cv2.line(img=img, pt1=self.pts_yellow[0], pt2=self.pts_yellow[2], color=(0, 255, 255), thickness=2)
        cv2.line(img=img, pt1=self.pts_yellow[1], pt2=self.pts_yellow[3], color=(0, 255, 255), thickness=2)

        return img
    
    def line_formula(self,pt1,pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        # eğim hesapla
        # nokta-eğim formülü kullanarak çizgilerin önce eğimini buluyoruz sonra eğim sayesinde kesişim noktasını buluyoruz. eğim(m), kesişim noktası(b) değerlerini döndürüyoruz
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)       # eğim bulmak için elimizdeki noktaları kullanıyoruz
        else:
            return (0,x1)  # Dikey çizgi (eğimin x ile kesiştiği nokta (x=c))

        b = y1 - m * x1     # "y = mx + b" eğim denklemini kullanarak kesişim (b) noktasını bulduk

        return (m,b)
    
    def humanDetection(self, image):
        img = image
        results = self.model.predict(img, verbose=False, classes=0, stream=True, save=False, boxes=False)
        red_right_line_a, red_right_line_b = self.line_formula(self.pts_red[1],self.pts_red[3])
        red_left_line_a, red_left_line_b = self.line_formula(self.pts_red[0],self.pts_red[2])

        yellow_right_line_a, yellow_right_line_b = self.line_formula(self.pts_yellow[1],self.pts_yellow[3])
        yellow_left_line_a, yellow_left_line_b = self.line_formula(self.pts_yellow[0],self.pts_yellow[2])

        red_flag = 0 # hem sarı hem kırmızı bölgede olup olmadığını kontrol eder
        if results:
            for result in results:
                numpy_list = result.boxes.numpy()
                for box in numpy_list:
                    x, y, w, h = box.xyxy[0].astype(int)
                    cv2.rectangle(img=img, pt1=(x,y), pt2=(w, h), color=(0, 0, 255), thickness=2)

                    # Kırmızı çizgiler için;
                    # İnsan sağdan gelirken durumu için;
                    # Sağdan gelen insan kutusu için (x,y+h) noktasını kullanacağız bu noktanın sağdaki kırmızı çizgiye temas ettiği yeri tespit ediyoruz. Noktaların algoritma mantığı için dosyaya paint görseli bıraktım.
                    if (x > min(self.pts_red[1][0],self.pts_red[3][0])): 

                        # # Hesaplanan x değeri, kırmızı sağ çizgi üzerinde belirli bir yükseklikteki noktanın x koordinatını verir.
                        x_line = int(((h) - red_right_line_b) / red_right_line_a)
                        if x_line > x:
                            cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                            red_flag = 1
                            self.warningStatus = 2
                        else:
                            self.warningStatus = 1
                        


                    # İnsan soldan gelirken durumu için;
                    # Soldan gelen insan kutusu için (x+w,y+h) noktasını kullanacağız bu noktanın soldaki kırmızı çizgiye temas ettiği yeri tespit ediyoruz
                    elif (w < max(self.pts_red[0][0],self.pts_red[2][0])):
                        # # Hesaplanan x değeri, kırmızı sol çizgi üzerinde belirli bir yükseklikteki noktanın x koordinatını verir.
                        x_line = int(((h) - red_left_line_b) / red_left_line_a)
                        if x_line < w:
                            cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                            red_flag = 1
                            self.warningStatus = 2
                        else:
                            self.warningStatus = 1
                    
                    else:
                        cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                        red_flag = 1
                        self.warningStatus = 2

                    # Sarı çizgiler için;
                    # Sağdan gelirken;
                    # Sağdan gelen insan kutusu için (x,y+h) noktasını kullanacağız bu noktanın sağdaki kırmızı çizgiye temas ettiği yeri tespit ediyoruz
                    if (x > min(self.pts_yellow[1][0],self.pts_yellow[3][0])): 

                        # # Hesaplanan x değeri, sarı sağ çizgi üzerinde belirli bir yükseklikteki noktanın x koordinatını verir.
                        x_line = int(((h) - yellow_right_line_b) / yellow_right_line_a)
                        if x_line > x:
                            if red_flag == 0:
                                cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                                self.warningStatus = 1
                        else:
                            self.warningStatus = 0
                        


                    # Soldan gelirken;
                    # Soldan gelen insan kutusu için (x+w,y+h) noktasını kullanacağız bu noktanın soldaki kırmızı çizgiye temas ettiği yeri tespit ediyoruz
                    elif (w < max(self.pts_yellow[0][0],self.pts_yellow[2][0])):
                        
                        # Hesaplanan x değeri, sarı sol çizgi üzerinde belirli bir yükseklikteki noktanın x koordinatını verir.
                        x_line = int(((h) - yellow_left_line_b) / yellow_left_line_a)
                        if x_line < w:
                            if red_flag == 0:
                                cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                                self.warningStatus = 1
                        else:
                            self.warningStatus = 0
                    
                    else:
                        if red_flag == 0:
                            cv2.putText(img=img, text=f"Warning {self.warningStatus}", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                            self.warningStatus = 1
                    

        return img, self.warningStatus

    def process(self):
        frame_count = 0
        while True:
            retval, img = self.video.read()
            frame_count += 1
            img = cv2.resize(src=img, dsize=(848, 480))
            
            if not retval:
                print("Frame Okunamıyor. Lütfen Kamera'nın Çalıştığından Emin Olduktan Sonra Tekrar Deneyin.")
                break
           
            try:
                img = self.drawTrapezoidal(image=img)
            except:
                print("Çizilen alan belirlenemedi!")
                
            
            try:
                img, self.warningStatus = self.humanDetection(image=img)
            except Exception as e:
                print(f"humanDetection fonksiyonu hatası. İnsan tespiti yapılamadı. {e}")
            
            cv2.imshow(winname="img", mat=img)
            cv2.waitKey(int(1000/self.cameraFPS))
    
    def __str__(self):
        self.video.release()
        cv2.destroyAllWindows()
    
p1 = TruckProtection()
p1.process()