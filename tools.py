import os
from typing import Generator, Optional
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image

from processing_params import CLIP, GRID, KSIZE, ALPHA, BETA, H_DENOISE, \
    TMP_WIN_SIZE, SEARCH_WIN_SIZE


class ImageProcessor:
    """
    Provides methods for reading, preprocessing, and filtering images for OCR.
    """

    def get_processed_image(self, img_path: str, params: dict) -> tuple[str, any]:
        """
        Processes an image from a given path using specified parameters.
        Returns the image filename and the processed image.
        """
        try:
            k_size = params.get('k_size', KSIZE)
            alpha = params.get('alpha', ALPHA)
            beta = params.get('beta', BETA)
            clip = params.get('clip', CLIP)
            grid = params.get('grid', GRID)
            h_denoise = params.get('h_denoise', H_DENOISE)
            tmp_win_size = params.get('tmp_win_size', TMP_WIN_SIZE)
            search_win_size = params.get('search_win_size', SEARCH_WIN_SIZE)

            loaded_img = self._read_img(img_path)
            cropped_img = self._crop_img(loaded_img)
            gray_img = self._convert_to_gray(cropped_img)
            blurred_img = self._gaussian_blur(gray_img, k_size)
            sharpened_img = self._add_weighted(gray_img, blurred_img, alpha, beta)
            enhanced_img = self._enhance_img(sharpened_img, clip, grid)
            denoised_img = self._denoise_img(enhanced_img, h_denoise,
                                             tmp_win_size, search_win_size)
            # Image.fromarray(
            #     denoised_img).show()  # uncomment if you want to see the processed photo
            return os.path.basename(img_path), denoised_img
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
    def _crop_img(loaded_img: np.ndarray) -> np.ndarray:
        """
        Crops the central region of the image based on fixed percentage ratios.
        """
        try:
            height, width = loaded_img.shape[:2]
            top = int(0.37 * height)
            bottom = int(0.64 * height)
            left = int(0.19 * width)
            right = int(0.82 * width)
            cropped_img = loaded_img[top:bottom, left:right]
        except Exception as e:
            raise RuntimeError(f"Error in _crop_img method: {e}") from e
        return cropped_img

    @staticmethod
    def _convert_to_gray(cropped_img: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale."""
        try:
            return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise RuntimeError(f"Error in _convert_to_gray: {e}") from e

    @staticmethod
    def _gaussian_blur(gray_img: np.ndarray, k: tuple[int, int]) -> np.ndarray:
        """Apply Gaussian blur to an image."""
        try:
            return cv2.GaussianBlur(gray_img, k, 0)
        except Exception as e:
            raise RuntimeError(f"Error in _gaussian_blur: {e}") from e

    @staticmethod
    def _denoise_img(img: np.ndarray, h: int | float, tmp_win_size: int, search_win_size: int) -> np.ndarray:
        """Apply denoising to an image."""
        try:
            return cv2.fastNlMeansDenoising(img, None, h=h,
                                           templateWindowSize=tmp_win_size,
                                           searchWindowSize=search_win_size)
        except Exception as e:
            raise RuntimeError(f"Error in _gaussian_blur: {e}") from e

    @staticmethod
    def _add_weighted(gray_img: np.ndarray, blurred_img: np.ndarray,
                      alpha: float, beta: float) -> np.ndarray:
        """Blend two images using weighted addition."""
        try:
            return cv2.addWeighted(gray_img, alpha, blurred_img, beta, 0)
        except Exception as e:
            raise RuntimeError(f"Error in _add_weighted: {e}") from e

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
    def _resize_for_ocr(img: np.ndarray) -> np.ndarray:
        """Resize image to improve OCR accuracy."""
        try:
            return cv2.resize(img, (0, 0), fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            raise RuntimeError(f"Error in _resize_for_ocr: {e}") from e

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
        results = reader.readtext(img)
        try:
            iccid_p1 = results[0][1] if iccid_no_p1 is None else iccid_no_p1
            iccid_p2 = results[1][1]
            iccid = iccid_p1 + iccid_p2
            # print(iccid)
            if self._checksum(iccid):
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
            return calculated_checksum == check_digit
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

    def update_csv(self, pcb_base: str, filename: str, iccid: str) -> bool:
        """
        Updates the 'pcbNumberSerial' column for rows matching the given ICCID.
        """
        pcb_number_serial = filename.rsplit(".", 1)[0] if pcb_base in filename else self._prepare_pcb_number_serial(pcb_base, filename)
        mask = self.df['ICCID'].str.contains(iccid, na=False)
        if mask.any():
            self.df.loc[mask, 'pcbNumberSerial'] = pcb_number_serial
            self.df.to_csv(self.csv_path, index=False, sep=';')
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
