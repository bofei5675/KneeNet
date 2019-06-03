import pydicom as dicom


def read_file(file):
    return dicom.read_file(file)