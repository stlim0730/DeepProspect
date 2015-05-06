import time

def get_time_string():
  return time.strftime("%H:%M:%S")

def timed_log(*args):
  args_str = [str(arg) for arg in args]
  concat_str = " ".join(args_str)
  print "#scrapper [" + get_time_string() + "] -", concat_str
  