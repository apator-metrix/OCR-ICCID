import os.path
import argparse
import sys
import warnings
from datetime import datetime

from processing_params import custom_params
from tools import ImageProcessor, ICCIDReader, CSVICCIDUpdater

# Disable warnings for pin_memory
warnings.filterwarnings("ignore", message=".*pin_memory.*",
                        category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="OCR ICCID -> CSV updater")

    parser.add_argument(
        "--path_to_images",
        type=str,
        required=True,
        help="Ścieżka do folderu ze zdjęciami"
    )

    parser.add_argument(
        "--path_to_csv",
        type=str,
        required=True,
        help="Ścieżka do pliku CSV z ICCID"
    )

    parser.add_argument(
        "--pcb_base",
        type=str,
        default="UN0651000.11-00-080403",
        help="Bazowy prefiks numeru PCB (np. do numerów plików)"
    )

    parser.add_argument(
        "--file_type",
        type=str,
        nargs='+',
        default=[".jpg", ".png", ".jpeg"],
        help="Lista akceptowanych typów plików, np. --file_type .jpg .png"
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Liczba zdjęć jaką chcemy sprawdzić"
    )

    parser.add_argument(
        "--use_reported_img",
        type=str,
        default=None,
        help="Wykonaj odczyt ICCID dla plików zawartych w wskazanym raporcie"
    )

    parser.add_argument(
        "--iccid_no_p1",
        type=str,
        help="Pierwsza część numeru ICCID (dla poprawy wyników jeśli pewna jest pierwsza część numeru)"
    )


    return parser.parse_args()


def main():
    args = parse_args()

    dt_now_start = datetime.now()
    dt_now_str = dt_now_start.strftime("%Y-%m-%d_%H-%M-%S")

    path_to_report = args.use_reported_img
    path_to_images = args.path_to_images
    path_to_csv = args.path_to_csv
    batch = args.batch

    pcb_base = args.pcb_base
    iccid_no_p1 = args.iccid_no_p1
    file_type = tuple(args.file_type)

    log_iccid_not_found_in_csv = "iccid_not_found_in_csv"
    log_unreadable_iccid = "iccid_unreadable"
    log_iccid_updated = "iccid_updated"

    image_processor = ImageProcessor()
    iccid_reader = ICCIDReader()
    csv_updater = CSVICCIDUpdater(path_to_csv)

    iter_image_paths = image_processor.iter_image_paths(path_to_images,
                                                        file_type)

    if path_to_report:
        reported_filenames = image_processor.load_reported_filenames(path_to_report)
        iter_image_paths = (path for path in iter_image_paths if
                       os.path.basename(path) in reported_filenames)

    iccid_processed_counter = 0
    iccid_update_counter = 0
    iccid_not_found_in_cdv_counter = 0
    iccid_unreadable_counter = 0
    iccid_readable_counter = 0

    num_of_images_to_process = batch if batch else image_processor.count_image_files(
        path_to_images, file_type) # do poprawy pod kątem batch i zdjec z raportu

    print("Rozpoczęcie przetwarzania zdjęć i aktualizowania pliku csv...")
    for idx, file in enumerate(iter_image_paths, 1):
        if batch:
            if idx > batch:
                break
        try:
            print(f"{idx}/{num_of_images_to_process}: {os.path.basename(file)}")
            iccid_processed_counter += 1
            iccid = None
            for params in custom_params.values():
                filename, processed_image = image_processor.get_processed_image(
                    file, params)
                iccid = iccid_reader.get_iccid(processed_image, iccid_no_p1)
                if iccid:
                    break
            if iccid:
                iccid_readable_counter += 1
                is_updated = csv_updater.update_csv(pcb_base, filename, iccid)
                if not is_updated:
                    iccid_not_found_in_cdv_counter += 1
                    csv_updater.logger(filename, dt_now_str,
                                       log_iccid_not_found_in_csv, iccid)
                else:
                    iccid_update_counter += 1
                    csv_updater.logger(filename, dt_now_str, log_iccid_updated, iccid)
            else:
                iccid_unreadable_counter += 1
                csv_updater.logger(filename, dt_now_str, log_unreadable_iccid)
        except Exception as e:
            print(f"Error while processing file {file}: {e}")

    dt_now_end = datetime.now()
    print("...zakończono")
    print(f"Czas trwania: {dt_now_end - dt_now_start}")
    print(f"PRZETWORZONYCH ZDJĘĆ: ", iccid_processed_counter)
    print(f"POPRAWNIE ODCZYTANE ICCID Z ZDJĘĆ: ", iccid_readable_counter)
    print(f"NIEPOPRAWNIE ODCZYTANE ICCID Z ZDJĘĆ: ", iccid_unreadable_counter)
    print(f"ZAKTUALIZOWANE POZYCJE W CSV: ", iccid_update_counter)
    print(f"NIEZAKTUALIZOWANE POZYCJE W CSV: ", iccid_not_found_in_cdv_counter)

if __name__ == "__main__":
    main()
