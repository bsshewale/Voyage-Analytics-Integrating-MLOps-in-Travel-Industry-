import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = None):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error(error_message, error_detail)

    def get_detailed_error(self, error_message, error_detail):
        if error_detail is None:
            return str(error_message)

        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"""
        Error occurred in script:
        File: {file_name}
        Line: {line_number}
        Message: {error_message}
        """

    def __str__(self):
        return self.error_message