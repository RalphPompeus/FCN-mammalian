#change raw_input to input

"""
1)look into Dario's function into FullyConvNet on GitHub to try and get the output graph for the loss
values
2) cross validation: 5 fold cross validation would be running the model 5 times and taking the averages
for all values
3) training set will require images of the same type - look to get 4/5 images, all cells are needed to 
be coloured in  
4) filters in fullyConvNet might have to be set to [16, 32, 64] in order to capture mammalian cells
filters like [2, 4, 8] might be insufficient for mammalian cells whose shapes vary significantly




"""

from FullyConvNet import FullyConvNet
import numpy as np
#Load or compile model and choose to run on CPU
M1 = FullyConvNet(512,512,False)
M1.defineModel()
M1.compileModel()

#Select Folders for training data answers and validation sets
folderData = "Files/Input/TrainingData/Data_Mamalian" #raw black and white images (8 bit greyscale not binary )
folderAnswers = "Files/Input/TrainingData/Answers_Mamalian" #threshold variation images (binary)
#folderValidate = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk" #any other raw black and white images to be tested
#   +",share=csce/biology/groups/pilizota/Leonardo_Castorina/"
#    + "RDM_TunnelSlide_Slide2_2_Compilled")




#folderData = "Files/TrainingSetFiles/NxN_originals"
#folderAnswers = "Files/TrainingSetFiles/NxN_binary"
folderValidate = "Files/TrainingSetFiles/NxN_Validation"


outputModelsFolder = "Files/Models/TempModels" #saves model here 





#Get training images and answers
trainingData, _ = M1.loadImagesFromFolder(prompt=False,path=folderData)
#trainingData = [trainingData[0]]


trainingData = [x[250:250+512,250:250+512] for x in trainingData] #might have to pad to 1024
trainingAnswers, _ = M1.loadImagesFromFolder(prompt=False,path=folderAnswers)
#trainingAnswers = [trainingAnswers[0]]
trainingAnswers = [x[250:250+512,250:250+512] for x in trainingAnswers]
#Load model training data

M1.loadTrainingData(trainingData,trainingAnswers,True)
#Train for a user defined number of epochs saving model and showing predictions
#each time
epochs = int(input("Epochs to train?"))
while epochs>0:
    M1.trainModel(batch_size=3,num_epochs=epochs)
    M1.saveModel("Files/Models/TempModels/MostRecent_FullyConvNetMamalian_1.h5")
    output = M1.predict(trainingData,threshold=False)
    shouldLoad = input("Ready to see data?(Y)")
    M1.displayData(trainingData,delay=0.1)
    M1.displayData(output,delay=2)
    epochs = int(input("Epochs to train?"))

#Test on validation data sets
validateData, _ = M1.loadImagesFromFolder(prompt=False,path=folderValidate)
output = M1.predict(validateData,threshold=True,thresh=0.5)

#Show inputand outputs interleaved
validationDisplayData = []
j=1
for i in range(len(validateData)*2-1):
    if i % 2 == 0:
        validationDisplayData.append(output[j-1])
        j += 1
    else:
        validationDisplayData.append(validateData[j-1])

M1.displayData(validationDisplayData,delay=2)

#M1.SaveImagesToFolder(predictions,imageNames,prompt="False", path="C:\\Users\\Ralph Pompeus\\Desktop\\MPhys_Project\\ImageSegmentationWithKeras-master\\ImageSegmentationWithKeras-master\\MachineSegmenter\\Files\\TrainingImages\\NxN_Validation",title="Select folder to save to.")
