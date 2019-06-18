import h5py


file_path ='/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/96m/9000798_96m_RIGHT_KNEE.hdf5'
f = h5py.File(file_path)

data = f['data'].value
print(data)