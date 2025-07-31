# OCR-ICCID


Narzędzie służące do automatycznego odczytu numerów ICCID ze zdjęć (np. etykiet) za pomocą OCR i aktualizacji pliku CSV z tymi danymi. Program przetwarza obrazy, wyciąga numery ICCID, weryfikuje ich poprawność i wstawia je do odpowiednich rekordów CSV, zapisując też logi z przebiegu operacji.
Funkcje:

 - Przetwarzanie zdjęć w podanym folderze lub na podstawie wskazanego raportu.

 - Odczyt ICCID z obrazów za pomocą EasyOCR.

 - Weryfikacja poprawności ICCID (algorytm Luhna).

 - Aktualizacja kolumny pcbNumberSerial w pliku CSV.

 - Tworzenie szczegółowych logów operacji.

### Wymagania:

- Python 3.12

- opencv-python

- easyocr

- numpy

- pandas

- Pillow

### Instalacja

Zainstaluj wymagane biblioteki korzystając z pliku requirements.txt:

    pip install -r requirements.txt

### Uruchomienie

    python main.py --path_to_images /home/mariusz/Pulpit/292025 --path_to_csv /home/mariusz/Pulpit/full_iccid.csv --batch 1

### Opcjonalne argumenty

    --file_type: lista rozszerzeń plików do przetwarzania (domyślnie .jpg .png .jpeg).

    --batch: liczba zdjęć do przetworzenia (np. --batch 100).

    --use_reported_img: ścieżka do pliku raportu z listą zdjęć do przeanalizowania, wyciągnię tylko zdjęcia będące na liście z ścieżki podaje w --path_to_images