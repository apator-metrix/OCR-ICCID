# OCR-ICCID

Narzędzie służące do automatycznego odczytu numerów ICCID ze zdjęć za pomocą biblioteki easyOcr i do aktualizacji numeru pcb w  pliku CSV. Program przetwarza obrazy, wyciąga numery ICCID, weryfikuje ich poprawność i wstawia numer pcb do odpowiednich rekordów CSV.
Funkcje:

 - Przetwarzanie zdjęć w podanym folderze

 - Odczyt ICCID z obrazów za pomocą EasyOCR.

 - Weryfikacja poprawności ICCID (algorytm Luhna).

 - Aktualizacja kolumny pcbNumberSerial w pliku CSV.

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

    python main.py --path_to_images /home/mariusz/Pulpit/292025 --path_to_csv /home/mariusz/Pulpit/full_iccid.csv

###  Argumenty
    --path_to_images: ścieżka do folderu z zdjęciami

    --path_to_csv: ścieżka do pliku csv, który ma być zaktualizowany

#### opcjonalne

    --file_type: lista rozszerzeń plików do przetwarzania (domyślnie .jpg .png .jpeg). (opcjo

    --batch: liczba zdjęć do przetworzenia (np. --batch 100).

    --iccid_no_p1: można podać pierwsze 10 znaków numeru ICCID (dla poprawy wyników jeśli ta wartość jest niezmienna)

    --use_reported_img: ścieżka do pliku raportu z listą zdjęć do przeanalizowania, wyciągnię tylko zdjęcia będące na liście z ścieżki podaje w --path_to_images