/* The measurer */
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "ClassFormatDataSet.h"
#include "OneHotClassFormat.h"
#include "TwoClassFormat.h"
#include "ClassMeasurer.h"

/* The trainer */
#include "StochasticGradient.h" //Use gradient descent
#include "MSECriterion.h" //Optimize mean square error

/* The MLP tools */
#include "ConnectedMachine.h"
#include "Linear.h"
#include "Tanh.h"
#include "Sigmoid.h"
#include "LogSoftMax.h"

/* Management of files */
#include "MatDataSet.h"
#include "DiskXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data

enum MlpMode{
	ALL,
	TRAIN,
	TEST
}; 

struct MlpParam{
	int n_inputs;
	int n_outputs;
	int n_hu;
	int max_iter; 
	real accuracy;
	real learning_rate;
	real weight_decay;
	char *file;
	char *valid_file;
	char *suffix;
	char *model_file;
};


using namespace Torch;


/* Prototype */
ConnectedMachine * createMachine(Allocator *allocator, MlpParam *param); 
StochasticGradient *createTrainer(Allocator *allocator, ConnectedMachine *mlp, MlpParam *opt=NULL); 
void Training(Allocator *allocator, MlpParam *param,CmdLine *cmd);
void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd);
real error(ConnectedMachine *mlp, MatDataSet *data);

int main(int argc, char **argv)
{

	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();
	MlpParam param;


	//=================== The command-line ==========================

	// Construct the command line
	CmdLine cmd;

	// Put the help line at the beginning
	//  cmd.info(help);

	// Train mode
	cmd.addMasterSwitch("--train");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("file", &(param.file), "the train file");
	cmd.addICmdArg("n_inputs", &(param.n_inputs), "input dimension of the data", true);
	cmd.addICmdArg("n_targets", &(param.n_outputs), "output dim. (regression) or # of classes (classification)", true);

	cmd.addText("\nLearning Options:");
	cmd.addICmdOption("-nhu", &(param.n_hu), 15, "number of hidden units", true);
	cmd.addICmdOption("-iter", &(param.max_iter), 25, "max number of iterations");
	cmd.addRCmdOption("-lr", &(param.learning_rate), 0.01, "learning rate");
	cmd.addRCmdOption("-e", &(param.accuracy), 1e-5, "end accuracy");
	cmd.addRCmdOption("-wd", &(param.weight_decay), 0, "weight decay", true);

	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-valid", &(param.valid_file),"NULL","the valid file");
	cmd.addSCmdOption("-save", &(param.model_file),"bp_model.dat","the model file");
	cmd.addSCmdOption("-suffix", &(param.suffix),"","Use a suffix for MSE (train/valid) error");

	cmd.addMasterSwitch("--test");
	cmd.addSCmdArg("model", &(param.model_file), "the model file");
	cmd.addSCmdArg("file", &(param.file), "the train file");
	cmd.addSCmdOption("-valid", &(param.valid_file),"NULL","the valid file");


	// Read the command line
	MlpMode mode = (MlpMode)cmd.read(argc, argv);

	//=================== Select the mode ==========================
	switch(mode) {
		case TRAIN:
			Training(allocator,&param,&cmd);
			break;
		case TEST:
			Testing(allocator,&param,&cmd);
			break;
		case ALL:
			printf("Not Implemented\n");
			break;
	}

	delete allocator;
	return(0);
}



void Training(Allocator *allocator, MlpParam *param, CmdLine *cmd) {

	//================== Check if we need to validate the data ===================
	bool validation(false);
	if(strcmp(param->valid_file,"NULL") != 0) validation=true;
	
	printf("Start Training ... \n");
	//=================== The Machine and its trainer  ===================	
	//Create Machine
	ConnectedMachine *mlp = createMachine(allocator,param);
	//Create Trainer
	StochasticGradient *trainer = createTrainer(allocator, mlp, param);


	//=================== DataSets  ===================
	//Create a data set
	DataSet *data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);
	//Computes means and standard deviation
	MeanVarNorm *mv_norm = new(allocator) MeanVarNorm(data);
	//Normalizes the data set
	data->preProcess(mv_norm);

	//=================== Mesurer  ===================
	// The list of measurers
	MeasurerList measurers;

	// The mean square error file on disk
	char mse_train_fname[256] = "MSE_train";
	strcat(mse_train_fname,param->suffix);
	DiskXFile *mse_train_file = new(allocator) DiskXFile(mse_train_fname, "w");
	MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, mse_train_file);
	measurers.addNode(mse_meas);

	//================= Validation (Data & Mesurer) ==============
	if(validation) {
		printf("Load Validation data...\n");
		char mse_valid_fname[256] = "MSE_valid";
		strcat(mse_valid_fname,param->suffix);
		DataSet *vdata =  new(allocator) MatDataSet(param->valid_file, param->n_inputs, param->n_outputs);
		vdata->preProcess(mv_norm);
		DiskXFile *mse_valid_file = new(allocator) DiskXFile(mse_valid_fname, "w");
		MSEMeasurer *mse_valid_meas = new(allocator) MSEMeasurer(mlp->outputs, vdata, mse_valid_file);
		measurers.addNode(mse_valid_meas);
	}


	//================= Train the MLP and find the weight ====================
    	trainer->train(data, &measurers);

	
      	//================ Save all the data in the model =====================
	DiskXFile model_(param->model_file, "w");
	cmd->saveXFile(&model_);
	mv_norm->saveXFile(&model_);
	mlp->saveXFile(&model_);
}

void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd) {

	//=================== Load Model =====================
	DiskXFile *model = new(allocator) DiskXFile(param->model_file, "r");
	cmd->loadXFile(model);


	//=================== Validation DataSets  ===================	
	printf("Load data...\n");
	printf("%s, %d, %d, %d\n",param->file,param->n_inputs,param->n_hu,param->n_outputs);
	MatDataSet *mat_vdata = new(allocator) MatDataSet(param->valid_file, param->n_inputs, param->n_outputs);
	MatDataSet *mat_data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);

	//=================== Normalize Data  ===================
	printf("Normalize data ...\n");
	MeanVarNorm *mv_norm = NULL;
	mv_norm = new(allocator) MeanVarNorm(mat_data);
	mv_norm->loadXFile(model); //Loading value from model to know how we normalize last time
        mat_data->preProcess(mv_norm);
	mat_vdata->preProcess(mv_norm);

	Sequence *class_labels = new(allocator) Sequence(2,1);
	class_labels->frames[0][0] =  0;
	class_labels->frames[0][1] =  1;

	DataSet *data = new(allocator) ClassFormatDataSet(mat_data, class_labels);
	DataSet *vdata = new(allocator) ClassFormatDataSet(mat_vdata, class_labels);
//    	OneHotClassFormat *class_format = new(allocator) OneHotClassFormat(1);
    	TwoClassFormat *class_format = new(allocator) TwoClassFormat(data);

	//=================== The Machine and its trainer  ===================	
	printf("Building Machine...\n");
	//Create Machine structure
	ConnectedMachine *mlp = createMachine(allocator,param);
	mlp->loadXFile(model); //Loading MLP weight structure
	//Create trainer structure
	StochasticGradient *trainer = createTrainer(allocator, mlp);

	//=================== Mesurer  ===================
	printf("Building Measurer...\n");
	// The list of measurers
	MeasurerList measurers;

	// Measurers on the training dataset
//	MSEMeasurer *test_mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, cmd->getXFile("testing_mse_err"));
//        measurers.addNode(test_mse_meas);
	ClassMeasurer *test_class_meas = new(allocator) ClassMeasurer(mlp->outputs, data, class_format, cmd->getXFile("test_class_err"));
	measurers.addNode(test_class_meas);

	// Measurers on the training dataset
//	MSEMeasurer *valid_mse_meas = new(allocator) MSEMeasurer(mlp->outputs, vdata, cmd->getXFile("validation_mse_err"));
//        measurers.addNode(valid_mse_meas);
	ClassMeasurer *valid_class_meas = new(allocator) ClassMeasurer(mlp->outputs, vdata, class_format, cmd->getXFile("valid_class_err"));
	measurers.addNode(valid_class_meas);

	printf("error :%f\n",error(mlp,mat_vdata));

	//=================== Test the machine ===============================
	printf("Test our built model\n");
	trainer->test(&measurers);
}
	

//=================== Create the MLP... =========================
ConnectedMachine *createMachine(Allocator *allocator, MlpParam *param) {
	ConnectedMachine *mlp = new(allocator) ConnectedMachine();
	if(param->n_hu > 0)  {
	
		//Set the first layer (input -> hidden units)
		Linear *c1 = new(allocator) Linear( param->n_inputs, param->n_hu);
		c1->setROption("weight decay", param->weight_decay);
		mlp->addFCL(c1);    
		
		//Set the second layer (threshold in hidden units)
		Tanh *c2 = new(allocator) Tanh(param->n_hu);
		mlp->addFCL(c2);

		//Set the third layer (Output value)
		Linear *c3 = new(allocator) Linear(param->n_hu, param->n_outputs);
		c3->setROption("weight decay", param->weight_decay);
		mlp->addFCL(c3);
	  
		//Put the last value to binary
		Sigmoid *c4 = new(allocator) Sigmoid(param->n_outputs);
      		mlp->addFCL(c4);
		}

	// Initialize the MLP
	mlp->build();
	mlp->setPartialBackprop();

	return mlp;
}


//=================== The Trainer ===============================
StochasticGradient *createTrainer(Allocator *allocator, ConnectedMachine *mlp, MlpParam *param) {

	printf("Trainer creation...\n");

	// The criterion for the StochasticGradient (MSE criterion)
	Criterion *criterion = NULL;
	criterion = new(allocator) MSECriterion(mlp->n_outputs);

	// The Gradient Machine Trainer
	StochasticGradient *trainer = new(allocator) StochasticGradient(mlp, criterion);

	if(param !=NULL) {
		trainer->setIOption("max iter",param->max_iter);
		trainer->setROption("end accuracy", param->accuracy);
		trainer->setROption("learning rate", param->learning_rate);
	}
	return trainer;
}

real error(ConnectedMachine *mlp, MatDataSet *data)
{    
    int failed=0;
    for (int i=0; i<data->n_examples; i++)
    {
        //printf("Iteration: %i\n",i);
        data->setExample(i);
        mlp->forward(data->inputs);
        Sequence *seq=mlp->outputs;
        //printf("%f , %f",*data->targets->frames[data->targets->frame_size-1],*seq->frames[seq->frame_size-1]);
        if (data->targets->frames[0][0]==0 && seq->frames[0][0]>.5) failed++;
        else if (data->targets->frames[0][0]==1 && seq->frames[0][0]<=.5) failed++;
        
    }
    return (real)failed/(real)data->n_examples;
}
