import os
import shutil
from typing import Generator, Optional
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image

from processing_params import CLIP, GRID, KSIZE


class ImageProcessor:
    """
    Provides methods for reading, preprocessing, and filtering images for OCR.
    """

    def get_processed_image(self, img_path: str, params: dict, img_package: str = 'blue') -> tuple[str, any]:
        """
        Processes an image from a given path using specified parameters.
        Returns the image filename and the processed image.
        """
        try:
            k_size = params.get('k_size', KSIZE)
            clip = params.get('clip', CLIP)
            grid = params.get('grid', GRID)

            loaded_img = self._read_img(img_path)

            if img_package == "silver":
                rotated_img = self._rotate_img(loaded_img, angle=2.5)
                cropped_img = self._crop_img(rotated_img, top_k=0.1, bottom_k=0.9, left_k=0.15, right_k=0.9)
                b, g, r = cv2.split(cropped_img)

                r = cv2.addWeighted(r, 1.3, r, 0, 0)
                g = cv2.addWeighted(g, 0.9, g, 0, 0)
                b = cv2.addWeighted(b, 3.0, b, 0, 0)

                color_modified = cv2.merge([b, g, r])

                gray_img = color_modified[:, :, 0]
                enhanced_img = self._enhance_img(gray_img, clip, grid)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                thicker = cv2.dilate(enhanced_img, kernel, iterations=1)
                thicker = cv2.erode(thicker, kernel2, iterations=1)
                blurred_img = self._blur(thicker, k_size)
                inverted = cv2.bitwise_not(blurred_img)
                Image.fromarray(inverted).show()
                return os.path.basename(img_path), inverted

            elif img_package == "blue":
                cropped_img = self._crop_img(loaded_img)
                gray_img = cropped_img[:, :, 0]
                enhanced_img = self._enhance_img(gray_img, clip, grid)
                blurred_img = self._blur(enhanced_img, k_size)
                Image.fromarray(blurred_img).show()
            # Image.fromarray(
            #     blurred_img).show()  # uncomment if you want to see the processed photo
                return os.path.basename(img_path), blurred_img
        except Exception as e:
            raise RuntimeError(
                f"Error in get_processed_image method: {e}") from e

    @staticmethod
    def iter_image_paths(folder_path: str, file_type: tuple[str]) -> Generator[str, None, None]:
        """
        Yields image file paths in a folder matching given file extensions.
        """
        try:
            with os.scandir(folder_path) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    if entry.is_file() and entry.name.endswith(tuple(file_type)):
                        yield entry.path
        except Exception as e:
            raise RuntimeError(f"Error in iter_image_paths method: {e}") from e

    @staticmethod
    def count_image_files(folder_path, file_type):
        """
        Counts the number of image files in a folder with specific extensions.
        """
        try:
            return sum(
                1
                for entry in os.scandir(folder_path)
                if entry.is_file() and entry.name.endswith(tuple(file_type))
            )
        except Exception as e:
            raise RuntimeError(f"Error in count_image_files method: {e}") from e

    @staticmethod
    def _read_img(img_path: str) -> any:
        """
        Reads an image from the specified path.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Error in _read_img, could not read image: {img_path}")
        return img

    @staticmethod
    def _crop_img(loaded_img: np.ndarray, top_k: float=0.37, bottom_k: float= 0.64, left_k: float = 0.19, right_k: float = 0.82) -> np.ndarray:
        """
        Crops the central region of the image based on fixed percentage ratios.
        """
        try:
            height, width = loaded_img.shape[:2]
            top = int(top_k * height)
            bottom = int(bottom_k * height)
            left = int(left_k * width)
            right = int(right_k * width)
            cropped_img = loaded_img[top:bottom, left:right]
        except Exception as e:
            raise RuntimeError(f"Error in _crop_img method: {e}") from e
        return cropped_img

    @staticmethod
    def _rotate_img(loaded_img: np.ndarray, angle: float = 0,
                    scale: float = 1) -> np.ndarray:
        try:
            h, w = loaded_img.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_img = cv2.warpAffine(loaded_img, m, (w, h))
        except Exception as e:
            raise RuntimeError(f"Error in _rotate_img method: {e}") from e
        return rotated_img

    @staticmethod
    def _convert_to_gray(cropped_img: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale."""
        try:
            return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise RuntimeError(f"Error in _convert_to_gray: {e}") from e

    @staticmethod
    def _blur(img: np.ndarray, k_size: int) -> np.ndarray:
        """Apply blur to an image."""
        try:
            return cv2.medianBlur(img, k_size)
        except Exception as e:
            raise RuntimeError(f"Error in _blur: {e}") from e

    @staticmethod
    def _enhance_img(sharpened_img: np.ndarray, clip: float,
                     grid: tuple[int, int]) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            return clahe.apply(sharpened_img)
        except Exception as e:
            raise RuntimeError(f"Error in _enhance_img: {e}") from e

    @staticmethod
    def reported_img_filter(file: str, report_file: str) -> bool:
        """Check if the file basename is listed in the report file."""
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            return os.path.basename(file) in file_content
        except Exception as e:
            raise RuntimeError(f"Error in reported_img_filter: {e}") from e

    @staticmethod
    def load_reported_filenames(report_file: str) -> set[str]:
        try:
            with open(report_file, "r", encoding="utf-8") as r_file:
                return {line.split("->")[0].strip() for line in r_file}
        except Exception as e:
            raise RuntimeError(f"Error loading report file: {e}") from e


class ICCIDReader:
    """
    Extracts and validates ICCID numbers from images using OCR and checksum verification.
    """

    def get_iccid(self, img: np.ndarray, iccid_no_p1) -> Optional[str]:
        """
        Extracts and returns a valid ICCID string from the given image.
        """
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img, allowlist="0123456789")
        try:
            confidence = results[1][-1] * 100
            iccid_p1 = results[0][1] if iccid_no_p1 is None else iccid_no_p1
            iccid_p2 = results[1][1]
            iccid = iccid_p1 + iccid_p2
            print(
                f"INFO ==============> {iccid} dopasowano z pewnością {round(confidence, 1)}% <===============")
            if self._checksum(iccid) and confidence >= 90:
                print(f"==============> {iccid} dopasowano z pewnością {round(confidence, 1)}% <===============")
                return iccid
        except Exception as e:
            print(f"Error in get_iccid: {e}")
            return None

    @staticmethod
    def _checksum(iccid: str) -> bool:
        """
        Validates the ICCID string using the Luhn algorithm.
        """
        if not iccid.isdigit() or len(iccid) != 19:
            raise ValueError("Error in _checksum method: The ICCID must contain exactly 19 digits.")
        try:
            digits = [int(d) for d in iccid[:-1]]
            check_digit = int(iccid[-1])

            total = 0
            reverse_digits = digits[::-1]

            for i, digit in enumerate(reverse_digits):
                if i % 2 == 0:
                    doubled = digit * 2
                    total += doubled if doubled < 10 else doubled - 9
                else:
                    total += digit

            calculated_checksum = (10 - (total % 10)) % 10
            checksum_matching = calculated_checksum == check_digit
            if not checksum_matching:
                print(f"Checksums mismatched for: {iccid}")
            return checksum_matching
        except Exception as e:
            print(f"Error in _checksum method: {e}")
            return False


class CSVICCIDUpdater:
    """
    Class for updating ICCID CSV files with PCB serial numbers.
    """
    def __init__(self, csv_path: str) -> None:
        """
        Initializes the object and loads data from a CSV file.
        """
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path, sep=';',
                                  dtype={'pcbNumberSerial': str})
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Failed to parse CSV file: {csv_path}\nDetails: {e}")
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while loading CSV: {csv_path}\nDetails: {e}")

    def create_new_csv_with_updated_rows(self, pcb_base: str, filename: str, iccid: str) -> bool:
        """
        Creates a new CSV file with updated 'pcbNumberSerial' column
        for rows matching the given ICCID.
        If the same pcbNumberSerial already exists in the output file,
        it won't be added again.
        """
        pcb_number_serial = filename.rsplit(".", 1)[0] if pcb_base in filename else self._prepare_pcb_number_serial(pcb_base, filename)
        mask = self.df['ICCID'].astype(str).str.contains(iccid, na=False)
        if mask.any():
            updated_rows = self.df.loc[mask].copy()
            updated_rows['pcbNumberSerial'] = pcb_number_serial
            output_path = f"{self.csv_path.rsplit('.', 1)[0]}_matched.csv"
            file_exists = os.path.isfile(output_path)
            if file_exists:
                existing_df = pd.read_csv(output_path, sep=';')
                if pcb_number_serial in existing_df['pcbNumberSerial'].astype(
                        str).values:
                    return False
            updated_rows.to_csv(output_path, index=False, sep=';', mode='a', header=not file_exists)
            return True
        return False

    @staticmethod
    def logger(filename: str, dt_now: str, log_type: str,
               iccid: str = "-" * 19) -> None:
        """
        Writes a log entry to a file named by date and log type.
        """
        try:
            with open(f"./reports/{dt_now}_logs_{log_type}.txt", "a") as file:
                file.write(
                    f"{filename} -> {iccid}\n")
        except IOError as e:
            print(f"Error writing to log: {e}")

    @staticmethod
    def _prepare_pcb_number_serial(base_num: str, filename: str) -> Optional[str]:
        """
        Creates a PCB serial number by combining base and part of filename.
        """
        try:
            pcb_num = base_num + "." + filename[4:].split(".")[0]
        except Exception as e:
            print(f"Error in _prepare_pcb_number_serial method: {filename} - {e}")
            return None
        return pcb_num


    @staticmethod
    def combine_csv_files(file1: str, file2: str) -> None:
        """
        Combines two CSV files and saves the result as 'full_iccid.csv'.
        """
        try:
            df1 = pd.read_csv(file1, sep=';')
            df2 = pd.read_csv(file2, sep=';')

            if 'pcbNumberSerial' not in df2.columns:
                df2['pcbNumberSerial'] = pd.NA
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            return
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
            return
        combined = pd.concat([df1, df2], ignore_index=True)
        try:
            combined.to_csv("full_iccid.csv", index=False, sep=';')
        except IOError as e:
            print(f"Error saving combined CSV file: {e}")


def move_images_based_on_report(report: str, start_path: str, end_path: str) -> None:
    """Copies images listed in the report file from start_path to end_path."""
    try:
        with open(report, "r", encoding="utf-8") as file:
            for line in file:
                img_name = line.split("->")[0].strip()
                full_path_img = os.path.join(start_path, img_name)
                if os.path.isfile(full_path_img):
                    shutil.copy(full_path_img, end_path)
                else:
                    print(f"File not found: {full_path_img}")
    except Exception as e:
        print(f"Error while moving images: {e}")

def get_num_of_updated_rows(csv_file: str) -> int:
    """Returns and prints the number of non-empty 'pcbNumberSerial' rows in the CSV file."""
    try:
        df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
        counts = df['pcbNumberSerial'].count()
        print(counts)
        return counts
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0

def get_report_updates_missing_in_csv(report_updated_rows: str, csv_file: str) -> list[str]:
    """Returns a list of image filenames from the report not found in any column of the CSV."""
    list_missing_positions = []
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", sep=';')
        with open(report_updated_rows, "r", encoding="utf-8") as file:
            for line in file:
                img_name = line.split("->")[0].strip()
                pcb = img_name.rsplit(".", 1)[0][4:]
                found = df.astype(str).apply(lambda col: col.str.contains(pcb, na=False)).any().any()
                if not found:
                    list_missing_positions.append(f"IS11{pcb}.jpg")
    except Exception as e:
        print(f"Error processing files: {e}")
    return list_missing_positions


def get_file_missing_in_csv(file_path: str, csv_file: str) -> list[str]:
    """Returns a list of image filenames from the directory not found in any column of the CSV."""
    list_missing_positions = []
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", sep=';')
        for filename in os.listdir(file_path):
            if filename.lower().endswith('.jpg'):
                pcb = filename.rsplit(".", 1)[0][4:]
                found = df.astype(str).apply(lambda col: col.str.contains(pcb, na=False)).any().any()
                if not found:
                    list_missing_positions.append(f"IS11{pcb}.jpg")
    except Exception as e:
        print(f"Error processing files: {e}")
    return list_missing_positions


def find_duplicates(report_updated_path):
    with open(report_updated_path, 'r') as file:
        text = file.read()

    with open(report_updated_path, 'r') as file2:

        for line in file2:
            iccid = line.split('->')[1].strip()
            count = text.count(iccid)
            if count > 1:
                print(iccid)





            # if img_package == "silver":
            #     rotated_img = self._rotate_img(loaded_img, angle=2.5)
            #     cropped_img = self._crop_img(rotated_img, top_k=0.1, bottom_k=0.9)
            #     b, g, r = cv2.split(cropped_img)
            #
            #     r = cv2.addWeighted(r, 1.3, r, 0, 20)  # wzmocnij czerwony
            #     g = cv2.addWeighted(g, 0.9, g, 0, 60)  # lekko osłab zielony
            #     b = cv2.addWeighted(b, 2.0, b, 0, 0)  # zostaw niebieski
            #
            #     color_modified = cv2.merge([b, g, r])
            #
            #     gray_img = color_modified[:, :, 0]
            #     enhanced_img = self._enhance_img(gray_img, clip, grid)
            #     blurred_img = self._blur(enhanced_img, k_size)
            #     Image.fromarray(color_modified).show()
            #     return os.path.basename(img_path), blurred_img