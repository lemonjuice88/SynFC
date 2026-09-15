# SynFC — Kritik Hata Notları

Bu dosya, ileride benzer bir hatayla tekrar karşılaşırsak "bunu daha önce
çözmüştük, sebebi buydu" diye hızlıca hatırlamak için tutuluyor. Yeni bir
kritik/sinsi hata bulunduğunda buraya eklenmeli.

---

## 1) Tarih formatı belirsizliği — `_estimate_contract_years` (finance_team.py)

**Ne oldu**: Transfermarkt'tan gelen sözleşme bitiş tarihi bazen
`30.06.2029` (nokta), bazen `30/06/2029` (eğik çizgi) formatında
geliyor. İlk yazdığımız kod sadece nokta formatını + ABD tipi
`%m/%d/%Y`'yi deniyordu, gerçek Avrupa formatı `%d/%m/%Y`'yi hiç
denemiyordu — bu yüzden bazı sorgularda tarih hiç okunamayıp
"veri yok" diyordu, halbuki veri gerçekten vardı.

**Kök sebep**: Format tahmin listesi eksikti, VE `%m/%d/%Y` /
`%d/%m/%Y` ikisi de bazı tarihlerde (örn. `05/06/2029`) **aynı anda
geçerli** olabiliyor, ikisi farklı sonuç verir (5 Haziran vs 6 Mayıs).
Hangisinin önce denendiği, belirsiz tarihlerde SESSİZCE yanlış
yorumlamaya yol açabilir.

**Çözüm**: `%d/%m/%Y` (gün/ay), `%m/%d/%Y`'den ÖNCE deneniyor artık —
çünkü Transfermarkt Avrupa merkezli, gün/ay çok daha olası gerçek
format. Eğer ileride hâlâ yanlış okunan bir tarih görürsen, önce BU
sıralamaya bak.

**Genel ders**: Tarih formatı varsayımları asla "tek format yeter"
diye düşünülmemeli — özellikle uluslararası kaynaklardan gelen veri,
birden fazla formatta gelebilir, VE format-tahmin sırası bazı
durumlarda sonucu SESSİZCE değiştirebilir (hata vermez, yanlış cevap
verir — bu daha tehlikeli).

---

## 2) "Sonuca bakarak yol tahmin etme" hatası — `_estimate_contract_years`

**Ne oldu**: Kod, bir sayının (`contract_years`) "gerçek veriden mi
yoksa varsayılan değerden mi" geldiğini, SONUCUN KENDİSİNE bakarak
(`== 3` mü diye) tahmin etmeye çalışıyordu. Ama gerçek bir sözleşme de
tesadüfen tam 3 yıl sürebilir — bu durumda kod, gerçek veriyi "varsayım"
diye yanlış etiketliyordu.

**Genel ders**: Bir değerin NASIL üretildiğini (gerçek mi tahmin mi),
üretilen DEĞERE bakarak asla çıkarma — bu iki şey birbirinden bağımsız.
Kaynak fonksiyon, bunu ayrı ve açık bir şekilde (bir flag/None dönerek)
bildirmeli.

**Son karar**: Varsayım tamamen kaldırıldı — veri yoksa artık hiç
tahmin üretilmiyor, direkt "veri yok" deniyor (kullanıcının açık
talebiyle, "kafana göre karar verme" prensibi).

---

## 3) Router'ın gevşek departman eşlemesi — `DEPARTMENT_SELECTOR_BILGI_SYSTEM` (engine.py)

**Ne oldu**: "Bir oyuncunun sözleşme/maaş/piyasa değeri" sorularını
hem `finance_team` hem `technical_team`'e yönlendiren bir kural vardı.
Ama "mali yük ne kadar" gibi SAF finansal bir soru, Teknik Ekip'in
kadro derinliği verisiyle hiç ilgili değil — gereksiz yere çağrılıp
"elimde veri yok" diye anlamsız bir cevap üretiyordu.

**Genel ders**: Router prompt'larında "X konusu -> A ve/veya B
departmanı" tarzı gevşek eşlemeler kurarken, gerçekten HER İKİ
departmanın da o soruya faydalı bir şey söyleyebileceğinden emin ol —
"ilgili görünüyor" yeterli değil, "gerçekten elinde veri var mı" sorusu
sorulmalı.

---

## 4) Erken yazılan metnin sonradan yanlışlanması — `accounting_node` (finance_team.py)

**Ne oldu**: Amortisman notunun sabit metni, "maaş verisi yok" varsayımıyla
yazılmıştı (`"no wage data available yet"`) — ama bu not, kod akışında
maaş verisi **henüz aranmadan önce** yazılıyordu. Maaş verisi az sonra
(aynı fonksiyonda) gerçekten bulunduğunda, bu erken yazılmış not
**yanlışlanmış** oluyordu — rapor hem "toplam €43.8M (maaş dahil)"
diyordu hem de "maaş dahil edilmedi" diyordu, aynı anda.

**Genel ders**: Bir metin parçası, kendisinden SONRA gelecek bir bilgiye
dair varsayımda bulunuyorsa (`"henüz yok"`, `"bilinmiyor"` gibi), o
varsayım kod akışının ilerleyen bir noktasında YANLIŞLANABİLİR. Çözüm:
metnin kendi KAPSAMINI belirtmesi ("bu rakam sadece X'i kapsar"),
başka bir bölümün DURUMU hakkında varsayımda bulunmaması ("Y verisi
yok" yerine "Y bölümüne bak, varsa").

---

## 5) Aynı hatanın alt katmanda tekrarı — `HEALTH_ROUTER_SYSTEM` (health_team.py)

**Ne oldu**: Ana motorun Router'ına "tam transfer kararında Health
varsayılan olarak dahil edilmeli" diye bir düzeltme yaptık (madde 3'ün
tersine, bu sefer "gevşetme" yönünde bir düzeltme). Ama Sağlık
departmanının **kendi içindeki** router'ı (`HEALTH_ROUTER_SYSTEM`),
hâlâ eski, dar kritere göre çalışıyordu — "konu metninde sakatlık/
fitness kelimesi açıkça geçmiyorsa Physio'yu seçme." Sonuç: ana motor
Sağlık'ı doğru çağırdı, ama Sağlık'ın İÇİNDEKİ Physio agent'ı hiç
çalışmadı, gerçek sakatlık verisi (`Thigh problems`, `Hip injury`)
hiç devreye girmedi — Club Doctor "elimde rapor yok" dedi, halbuki
veri tool seviyesinde gerçekten vardı ve doğru çekiliyordu.

**Genel ders**: Bir "varsayılan davranış" (default) kuralını bir
katmanda (ana Router) düzelttiğinde, AYNI mantığın gerektiği diğer
katmanları (departmanların kendi iç router'ları gibi) da kontrol et.
Bir üst katmanın doğru karar vermesi, alt katmanların da otomatik
doğru davranacağı anlamına gelmez — her seviyenin kendi, bağımsız
karar mantığı var, biri düzelince diğeri kendiliğinden düzelmiyor.

**Teşhis ipucu**: Director/sentezleyici agent'ın "elimde X raporu yok"
demesi, mutlaka "tool başarısız oldu" anlamına gelmez — bazen "o
agent hiç çalıştırılmadı" anlamına gelir (iç router onu seçmemiştir).
Bu ikisini ayırt etmek için, önce tool'un kendisini (izole) test et,
sonra iç router'ın o agent'ı gerçekten seçip seçmediğini kontrol et.

---

## Genel prensip (hepsinin ortak noktası)

Bugünkü üç hatanın üçü de aynı kökten geliyor: **prompt/kod yazarken
"muhtemelen doğrudur" diye ilerlemek, gerçek veriyle test etmeden.**
Sistem mesajlarını (ROUTER_SYSTEM, departman persona'ları) satır satır
okuma turu, tam olarak bu tür sinsi hataları önceden yakalamak için.

---

## ÖNCELİKLİ GÖREV — Kaggle veri setinin güncelliği (2024/25 → 2025/26)

**Sorun**: `player_stats.sqlite` ve ilgili CSV'ler (Kaggle'dan indirilen)
şu an **2024/25 sezonu** verisiyle sabit. SQL sorguları (Data Scientist),
squad/istatistik context'leri (Technical Analyst, Transfer Analyst,
Scout) — hepsi bu sabit veriye dayanıyor. Sezon ilerledikçe bu veri
**eskiyecek.**

**Yapılması gereken**: Ya (A) Kaggle dataset'inin 2025/26 güncellemesini
periyodik olarak indirip `data/player_stats.sqlite`'ı yeniden kurmak
(hatırla: dosya `if not os.path.exists` ile kontrol ediliyor, güncel
veri için eski `.sqlite` dosyasının SİLİNMESİ gerekiyor), ya da (B) canlı
bir kaynağa geçmek (daha zor, ama otomatik güncel kalır).

**Öncelik**: Kullanıcı bunu "hafife almama" diye özellikle vurguladı —
release öncesi (20 Eylül) ele alınmalı.

---

## 6) Perplexity'nin "canlı" aramasının eski haberleri güncelmiş gibi sunması

**Ne oldu**: Ernest Poku/Beşiktaş test senaryosunda, Reporter agent'ı
"Solskjaer'in istediği bir profile benziyor" dedi — ama Solskjaer,
**2025 Ağustos'ta** (Conference League elemesinden sonra) Beşiktaş'tan
kovulmuş, yerine Sergen Yalçın gelmiş, hatta bir kaynağa göre Yalçın da
**2026 Mayıs'ta** kovulmuş. Yani "canlı arama", neredeyse **1 yıllık
bayat bir haberi**, hiçbir tazelik uyarısı olmadan güncelmiş gibi
sundu.

**Kök sebep (kesinleşmedi, ama en olası)**: Perplexity'nin arama
sonuçları, muhtemelen o dönemde (Solskjaer henüz görevdeyken) yazılmış
eski makaleleri de içeriyor, ve `SEARCH_SYSTEM_PROMPT`
(`media_web_search_perplexity.py`), modele **"bugünün tarihini"**
vermiyor, personel/görev iddialarının (kimin teknik direktör/başkan
olduğu gibi) **güncelliğini doğrulamasını** hiç istemiyor.

**Yapılması gereken**: `SEARCH_SYSTEM_PROMPT`'a bugünün tarihini
enjekte edip, "personel/görev iddialarının hâlâ geçerli olduğunu
doğrula, eski bir makaleden geliyorsa açıkça belirt" diye bir talimat
eklemek. Bu, henüz uygulanmadı — sonraki oturumda ele alınmalı.

**Öncelik**: Release öncesi (20 Eylül) ele alınmalı — özellikle Medya
departmanının güvenilirliğini doğrudan etkiliyor.