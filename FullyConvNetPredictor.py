from FullyConvNet import FullyConvNet

#Set up and load network using GPU
M1 = FullyConvNet(512,512,True)



#Get Model
#my model from training data 
#M1.loadModel(prompt=False,path="C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\Models\\TempModels\\MostRecent_FullyConvNetMamalian_1.h5")

#saved Mammalian cell model
#M1.loadModel(prompt=False,path="C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\Models\\TempModels\\MostRecent_FullyConvNetMamalian_100Epochs_8_16_32.h5")


############ 
M1.loadModel(prompt=False,path="C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\Models\\TempModels\\MostRecent_FullyConvNetMamalian_100Epoch_16_32_64.h5")
############


#Select folder to analyse
#Change path to your images you want to validate 
#images, imageNames = M1.loadImagesFromFolder(prompt=False,  path= "C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\Input\\TrainingData\\Data_Mamalian", title= "Select folder to analyse.")



#==============================================================================


#my path for the files I want to find - validation file path for network image improvement reference
images, imageNames = M1.loadImagesFromFolder(prompt=False,  path= "C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\cellRun3", title= "Select folder to analyse.")



#==============================================================================


######### validaiton for Teuta
#images, imageNames = M1.loadImagesFromFolder(prompt=False,  path= "C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\cellRun3", title= "Select folder to analyse.")
#########


images = [x[250:250+512,250:250+512] for x in images] #might have to pad to 1024
#uncomment this if needs be ^^^



M1.displayData(images)

#Predict
predictions = M1.predict(images)
M1.displayData(predictions)

#Save images
M1.SaveImagesToFolder(predictions,imageNames,prompt=False, path="C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\Codes\\Comparison", title="Select folder to save to.")
