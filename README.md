# yoloV8_cizgi_ve_insanTespiti
YoloV8 hazır model kullanarak çizdiğimiz sınırlara insan girdiğinde uyarı veren bir sistem yaptım.

Kırmızı çizginin sağından gelirken dikdörtgenin sağ alt köşesini baz alıyoruz ve kırmızı çizgi üzerinden eğim ile çizgiye temas olan x noktasını buluyoruz aralarında x karşılaştırması yapıyoruz (insanX>çizgiX) 
ve bu sonuca göre bir uyarı sağlıyoruz.

Diğer çizgilerde de benzer mantık söz konusu oluyor. Soldan gelirken ise dikdörtgenin sağ alt noktasını baz alarak aynı işlemleri yapıyoruz. 
Paint çizimine bakabilirsiniz.
