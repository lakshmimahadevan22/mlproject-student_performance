import traceback
from src.logger import logging

def error_message_detail(error):
    tb = traceback.TracebackException.from_exception(error)
    tb_stack = tb.stack[-1]  # Get the last call in the stack trace
    file_name = tb_stack.filename
    line_number = tb_stack.lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        self.error_message = error_detail

    def __str__(self):
        return f"{self.message}: {self.error_detail}"

try:
    # Code that might raise an exception
   # raise ValueError("An example error")
   pass
except Exception as e:
    error_detail = error_message_detail(e)
    raise CustomException(error_detail)


#if __name__=="__main__":

 #   try:
  #      a=1/0
   # except Exception as e:
    #    logging.info("Divide by Zero")
     #   raise CustomException(e)
    
