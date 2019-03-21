import argparse
import pandas as pd
import os
from pathlib import Path
import dicom
import numpy 
import h5py 

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = '/gpfs/data/denizlab/Datasets/OAI_original', type=str,
                        dest='input', help='folder to read dicom images')
    parser.add_argument('--output', default = '/gpfs/data/denizlab/Datasets/OAI', type=str,
                        dest='output', help='folder to output h5 images')
    parser.add_argument('-y', '--year', default=['96m'], nargs='+', help="list of years to convert")
    parser.add_argument('--name', default='SAG_IW_TSE', type=str, dest='contrast', help="sequnce name to use")

    return parser.parse_args()



#imagesFolder = 'R:\denizlab\denizlabspace\Datasets\OAI\images'
#outFolder_base = 'R:\denizlab\denizlabspace\Datasets\OAI_h5\SAG_3D_DESS'
#outFolder_base = 'Z:\Datasets\OAI\SAG_3D_DESS'
#year = ['00m','12m','18m','24m','30m','36m','48m','72m','96m']

def convertTo_h5(args):
    imagesFolder = args.input
    outFolder_base = args.output
    year = args.year
    contrast = args.contrast
    print(imagesFolder,outFolder_base,year,contrast)

    for yi in range(len(year)):
        outFolder = Path(outFolder_base,contrast,year[yi])
        if outFolder.exists() ==False:
            os.mkdir(str(outFolder))
        df_save = pd.DataFrame(columns =['FileName', 'Month', 'ParticipantID','StudyDate','Side', 
                                 'SeriesDescription',  'MatrixSize', 
                                'VoxelSize','Folder'])
        df = pd.read_csv(Path(imagesFolder,year[yi],'contents.csv'))

        df_lr = df[df['SeriesDescription'].isin([contrast+'_RIGHT', contrast+'_LEFT', contrast+'_LEFT '])]

        for fi in range(len(df_lr)):
            PathDicom = Path(imagesFolder,year[yi],(df_lr.iloc[fi]['Folder']))
            print('year: %s, fi:%d/%d %s'%(year[yi], fi,len(df_lr),str(PathDicom)))

            lstFilesDCM = []  # create an empty list
            for fileList in list(PathDicom.glob('**/*')):
                lstFilesDCM.append(fileList)
            lstFilesDCM = sorted(lstFilesDCM)
            # Get ref file
            RefDs = dicom.read_file(str(lstFilesDCM[0]))

            # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

            # Load spacing values (in mm)
            ConstPixelSpacing = '%.3fx%.3fx%.1f'%(float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

            # The array is sized based on 'ConstPixelDims'
            ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
            
            #if len(lstFilesDCM) !=160:
            #    print('Number of slices % d is different than 160 in %s_%s_%s_%s.hdf5'%(len(lstFilesDCM),df_lr.iloc[fi]['ParticipantID'],year[yi],
            #        df_lr.iloc[fi]['SeriesDescription'][len(contrast)+1:],df_lr.iloc[fi]['SeriesDescription'][:len(contrast)]))
            # loop through all the DICOM files
            lidx=0
            for filenameDCM in lstFilesDCM:
                # read the file
                ds = dicom.read_file(str(filenameDCM))
                # store the raw image data
                cem=str(filenameDCM)
                #print(cem, int(cem[-3:])-1,lidx)
                ArrayDicom[:, :, lidx] = ds.pixel_array
                lidx+=1
                #if len(lstFilesDCM) != 160:    
                #    ArrayDicom[:, :, int(cem[-3:])-1-(160-len(lstFilesDCM))] = ds.pixel_array
                #else:
                #    ArrayDicom[:, :, int(cem[-3:])-1] = ds.pixel_array
                # update the dataframe
            file_out = '%s_%s_%s_%s.hdf5'%(df_lr.iloc[fi]['ParticipantID'],year[yi],
                                        df_lr.iloc[fi]['SeriesDescription'][len(contrast)+1:],df_lr.iloc[fi]['SeriesDescription'][:len(contrast)])
            df_save = df_save.append({'FileName' : file_out,
                                'Month': year[yi], 
                                'ParticipantID' : df_lr.iloc[fi]['ParticipantID'], 
                                'Folder' :  df_lr.iloc[fi]['Folder'],
                                'StudyDate': df_lr.iloc[fi]['StudyDate'], 
                                'SeriesDescription':df_lr.iloc[fi]['SeriesDescription'],
                                'Side': df_lr.iloc[fi]['SeriesDescription'][len(contrast)+1:], 
                                'MatrixSize':'%dx%dx%d'%(int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM)), 
                                'VoxelSize': ConstPixelSpacing},ignore_index=True)


            f = h5py.File(str(Path(outFolder,file_out)),'w')
            f.create_dataset('data', data = ArrayDicom)
            f.create_dataset('PixelDims', data = ConstPixelDims)
            f.create_dataset('PixelSpacing', data = ConstPixelSpacing)
            f.create_dataset('Folder', data = df_lr.iloc[fi]['Folder'])
            f.close()
            #if fi%500==0:

        df_save.to_pickle(str(Path(outFolder,'HDF5_File_Info_%s.pkl'%year[yi])))
        df_save.to_csv(str(Path(outFolder,'HDF5_File_Info_%s.csv'%year[yi])),index=False)
    print('DONE')

if __name__ == '__main__':
    args = parse()
    convertTo_h5(args)