# I/O with gz compressed files
# https://stackoverflow.com/a/13083308

import gzip

def write_gz(filepath, file_content):
    file = gzip.open(filepath, 'wt')
    file.write(file_content)
    file.close()

def read_gz(filepath):
    file = gzip.open(filepath, 'rt')
    file_content = file.read()
    return file_content
