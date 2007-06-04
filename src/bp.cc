//#include "ClassFormatDataSet.h"
//#include "OneHotClassFormat.h"

/* The measurer */
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"

/* The trainer */
#include "StochasticGradient.h" //Use gradient descent
#include "MSECriterion.h" //Optimize mean square error

/* The MLP tools */
#include "ConnectedMachine.h"
#include "Linear.h"
#include "Tanh.h"
#include "LogSoftMax.h"

/* Management of files */
#include "MatDataSet.h"
#include "DiskXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data

enum MlpMode{
	ALL,
	TRAIN,
	VALID,
	TEST
}; 

class LearningOpt{

public:
	int max_iter; 
	real accuracy;
	real decay;
	real learning_rate; 
	real weight_decay;

	LearningOpt() : 
		max_iter(500), accuracy(0.00001),decay(0),learning_rate(0.01),weight_decay(0)
	{}

};


using namespace Torch;


/* Prototype */
ConnectedMachine createMachine(Allocator *allocator, int n_inputs, int n_hu, int n_outputs); 
StochasticGradient createTrainer(Allocator *allocator, ConnectedMachine *mlp, LearningOpt *opt, int n_outputs);


void Training(Allocator *allocator, char *file, char *model_file, int n_inputs, int n_hu, int m_outputs, LearningOpt *opt,CmdLine *cmd);
void Validation(Allocator *allocator, char *valid_file, DiskXFile *model, CmdLine *cmd); 


int main(int argc, char **argv)
{
	char *file;

	int n_inputs;
	int n_targets;
	int n_hu;

	char *model_file;

	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();

	LearningOpt learn_opt;



	//=================== The command-line ==========================

	// Construct the command line
	CmdLine cmd;

	// Put the help line at the beginning
	//  cmd.info(help);

	// Train mode
	cmd.addMasterSwitch("--train");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("file", &file, "the train file");
	cmd.addICmdArg("n_inputs", &n_inputs, "input dimension of the data", true);
	cmd.addICmdArg("n_targets", &n_targets, "output dim. (regression) or # of classes (classification)", true);

	cmd.addText("\nModel Options:");
	cmd.addICmdOption("-nhu", &n_hu, 15, "number of hidden units", true);
	cmd.addSCmdOption("-model", &model_file,"mlp_model.dat","the model file");

	cmd.addMasterSwitch("--valid");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &model_file, "the model file");
	cmd.addSCmdArg("file", &file, "the train file");


	cmd.addMasterSwitch("--test");
	cmd.addSCmdArg("model", &model_file, "the model file");
	cmd.addSCmdArg("file", &file, "the train file");


	// Read the command line
	MlpMode mode = (MlpMode)cmd.read(argc, argv);
	DiskXFile *model = NULL;


	switch(mode) {
		case TRAIN:
			Training(allocator,file,model_file,n_inputs,n_hu,n_targets,&learn_opt,&cmd);
			break;
		case VALID:
			model = new(allocator) DiskXFile(model_file, "r");
			Validation(allocator,file,model,&cmd);
			break;
		case TEST:
			model = new(allocator) DiskXFile(model_file, "r");
			cmd.loadXFile(model);
			/* //Test training
			    trainer.test(&measurers);
			*/
			break;
		case ALL:
			printf("Not Implemented\n");
			break;
	}

	delete allocator;
	return(0);
}



void Training(Allocator *allocator, char *file, char *model_file, int n_inputs, int n_hu, int n_outputs, LearningOpt *learn_opt,CmdLine *cmd) {

	printf("Start Training ... \n");

	//=================== The Machine and its trainer  ===================	
	//Create Machine
	ConnectedMachine mlp = createMachine(allocator, n_inputs,n_hu,n_outputs);

	//Create Trainer
	StochasticGradient trainer = createTrainer(allocator, &mlp, learn_opt, n_outputs);


	//=================== DataSets  ===================
	//Create a data set
	DataSet *data = new(allocator) MatDataSet(file, n_inputs, n_outputs);
	//Computes means and standard deviation
	MeanVarNorm *mv_norm = new(allocator) MeanVarNorm(data);
	//Normalizes the data set
	data->preProcess(mv_norm);


	//=================== Mesurer  ===================
	// The list of measurers
	MeasurerList measurers;

	// The mean square error file on disk
	DiskXFile *mse_train_file = new(allocator) DiskXFile("the_mse_err", "w");
	MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp.outputs, data, mse_train_file);
	measurers.addNode(mse_meas);


//=================== Let's go... ===============================
    	trainer.train(data, &measurers);
      
	DiskXFile model_(model_file, "w");
	cmd->saveXFile(&model_);
	mv_norm->saveXFile(&model_);
	mlp.saveXFile(&model_);
}


void Validation(Allocator *allocator, char *valid_file, DiskXFile *model, CmdLine *cmd) {

	int n_inputs;
	int n_hu;
	int n_targets;

	printf("Load model...\n");
	//=================== Model built  ===================
	//Load the model from validation set to obtain old normalization
	MeanVarNorm *mv_norm = NULL;

	cmd->loadXFile(model);
	mv_norm->loadXFile(model);


	printf("Starting Validation\n");
	//=================== The Machine and its trainer  ===================	
	//Create Machine
	ConnectedMachine mlp;
  	mlp.build();
 	mlp.setPartialBackprop();
	mlp.loadXFile(model);

	//Create Trainer To be clear
	//StochasticGradient trainer = createTrainer(allocator, &mlp);


	printf("Load data...\n");
	//=================== Validation DataSets  ===================	
	DataSet *valid_data = new(allocator) MatDataSet(valid_file, n_inputs, n_targets, false, -1, true);
        valid_data->preProcess(mv_norm);

	//=================== Mesurer  ===================
	// The list of measurers
	MeasurerList measurers;

	// Measurers on the training dataset
	MSEMeasurer *valid_mse_meas = new(allocator) MSEMeasurer(mlp.outputs, valid_data, cmd->getXFile("the_valid_mse_err"));
        measurers.addNode(valid_mse_meas);

	printf("let's go...\n");
	//=================== Let's go... ===============================
//	trainer.test(&measurers);
}

	

//=================== Create the MLP... =========================
ConnectedMachine createMachine(Allocator *allocator, int n_inputs, int n_hu, int n_outputs) {

	ConnectedMachine mlp;
	if(n_hu > 0)  {
	
		//Set the first layer (input -> hidden units)
		Linear *c1 = new(allocator) Linear(n_inputs, n_hu);
		//  c1->setROption("weight decay", weight_decay);
		mlp.addFCL(c1);    
		
		//Set the second layer (threshold in hidden units)
		Tanh *c2 = new(allocator) Tanh(n_hu);
		mlp.addFCL(c2);

		//Set the last layer (Output value)
		Linear *c3 = new(allocator) Linear(n_hu, n_outputs);
		// c3->setROption("weight decay", weight_decay);
		mlp.addFCL(c3);
		}

	// Initialize the MLP
	mlp.build();
	mlp.setPartialBackprop();

	return mlp;
}


//=================== The Trainer ===============================
StochasticGradient createTrainer(Allocator *allocator, ConnectedMachine *mlp, LearningOpt *opt, int n_outputs) {

	// The criterion for the StochasticGradient (MSE criterion)
	Criterion *criterion = NULL;
	criterion = new(allocator) MSECriterion(n_outputs);

	// The Gradient Machine Trainer
	StochasticGradient trainer(mlp, criterion);

	trainer.setIOption("max iter",opt->max_iter);
	trainer.setROption("end accuracy", opt->accuracy);
	trainer.setROption("learning rate", opt->learning_rate);
	trainer.setROption("learning rate decay", opt->decay);

	return trainer;
}


