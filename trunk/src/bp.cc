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
	TEST
}; 

class LearningOpt{

public:
	int max_iter; 
	real accuracy;
	real decay;
	real learning_rate; 
	real weight_decay;
	char *valid_file;
	char *suffix;

	LearningOpt() : 
		max_iter(500), accuracy(0.00001),decay(0),learning_rate(0.01),weight_decay(0),valid_file(NULL),suffix("")
	{}

};


using namespace Torch;


/* Prototype */
ConnectedMachine * createMachine(Allocator *allocator, int n_inputs, int n_hu, int n_outputs); 
StochasticGradient *createTrainer(Allocator *allocator, ConnectedMachine *mlp, LearningOpt *opt=NULL); 
void Training(Allocator *allocator, char *file, char *model_file, int n_inputs, int n_hu, int m_outputs, LearningOpt *opt,CmdLine *cmd);
void Testing(Allocator *allocator, char *valid_file, DiskXFile *model, int n_inputs, int n_hu, int n_targets, CmdLine *cmd);

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

	cmd.addText("\nLearning Options:");
	cmd.addICmdOption("-nhu", &n_hu, 15, "number of hidden units", true);
	cmd.addICmdOption("-iter", &(learn_opt.max_iter), 25, "max number of iterations");
	cmd.addRCmdOption("-lr", &(learn_opt.learning_rate), 0.01, "learning rate");
	cmd.addRCmdOption("-e", &(learn_opt.accuracy), 0.00001, "end accuracy");
	cmd.addRCmdOption("-lrd", &(learn_opt.decay), 0, "learning rate decay");
	cmd.addRCmdOption("-wd", &(learn_opt.weight_decay), 0, "weight decay", true);

	cmd.addText("\nMisc Options:");
	cmd.addSCmdOption("-valid", &(learn_opt.valid_file),"NULL","the valid file");
	cmd.addSCmdOption("-model", &model_file,"bp_model.dat","the model file");
	cmd.addSCmdOption("-suffix", &(learn_opt.suffix),"","Use a suffix for MSE (train/valid) error");

	cmd.addMasterSwitch("--test");
	cmd.addSCmdArg("model", &model_file, "the model file");
	cmd.addSCmdArg("file", &file, "the train file");


	// Read the command line
	MlpMode mode = (MlpMode)cmd.read(argc, argv);
	DiskXFile *model = NULL;


	//=================== Select the mode ==========================
	switch(mode) {
		case TRAIN:
			Training(allocator,file,model_file,n_inputs,n_hu,n_targets,&learn_opt,&cmd);
			break;
		case TEST:
			model = new(allocator) DiskXFile(model_file, "r");
			cmd.loadXFile(model);
			Testing(allocator,file,model,n_inputs,n_hu,n_targets,&cmd);
			break;
		case ALL:
			printf("Not Implemented\n");
			break;
	}

	delete allocator;
	return(0);
}



void Training(Allocator *allocator, char *file, char *model_file, int n_inputs, int n_hu, int n_outputs, LearningOpt *learn_opt,CmdLine *cmd) {

	//================== Check if we need to validate the data ===================
	bool validation(false);
	if(strcmp(learn_opt->valid_file,"NULL") != 0) validation=true;
	
	printf("Start Training ... \n");
	//=================== The Machine and its trainer  ===================	
	//Create Machine
	ConnectedMachine *mlp = createMachine(allocator, n_inputs,n_hu,n_outputs);
	//Create Trainer
	StochasticGradient *trainer = createTrainer(allocator, mlp, learn_opt);


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
	char mse_train_fname[256] = "MSE_train";
	strcat(mse_train_fname,learn_opt->suffix);
	DiskXFile *mse_train_file = new(allocator) DiskXFile(mse_train_fname, "w");
	MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, mse_train_file);
	measurers.addNode(mse_meas);

	//================= Validation (Data & Mesurer) ==============
	if(validation) {
		printf("Load Validation data...\n");
		char mse_valid_fname[256] = "MSE_valid";
		strcat(mse_valid_fname,learn_opt->suffix);
		DataSet *vdata =  new(allocator) MatDataSet(learn_opt->valid_file, n_inputs, n_outputs);
		vdata->preProcess(mv_norm);
		DiskXFile *mse_valid_file = new(allocator) DiskXFile(mse_valid_fname, "w");
		MSEMeasurer *mse_valid_meas = new(allocator) MSEMeasurer(mlp->outputs, vdata, mse_valid_file);
		measurers.addNode(mse_valid_meas);
	}


	//================= Train the MLP and find the weight ====================
    	trainer->train(data, &measurers);

	
      	//================ Save all the data in the model =====================
	DiskXFile model_(model_file, "w");
	cmd->saveXFile(&model_);
	mv_norm->saveXFile(&model_);
	mlp->saveXFile(&model_);
}


void Testing(Allocator *allocator, char *file, DiskXFile *model, int n_inputs, int n_hu, int n_targets, CmdLine *cmd) {


	//=================== Validation DataSets  ===================	
	printf("Load data...\n");
	printf("%s, %d, %d, %d",file,n_inputs,n_hu,n_targets);
	DataSet *data = new(allocator) MatDataSet(file, n_inputs, n_targets);

	//=================== Normalize Data  ===================
	printf("Normalize data ...\n");
	MeanVarNorm *mv_norm = NULL;
	mv_norm = new(allocator) MeanVarNorm(data);
	mv_norm->loadXFile(model); //Loading value from model to know how we normalize last time
        data->preProcess(mv_norm);

	//=================== The Machine and its trainer  ===================	
	printf("Building Machine...\n");
	//Create Machine structure
	ConnectedMachine *mlp = createMachine(allocator,n_inputs,n_hu,n_targets);
	mlp->loadXFile(model); //Loading MLP weight structure
	//Create trainer structure
	StochasticGradient *trainer = createTrainer(allocator, mlp);

	//=================== Mesurer  ===================
	printf("Building Measurer...\n");
	// The list of measurers
	MeasurerList measurers;

	// Measurers on the training dataset
	MSEMeasurer *test_mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, cmd->getXFile("testing_mse_err"));
        measurers.addNode(test_mse_meas);

	//=================== Test the machine ===============================
	printf("Test our built model\n");
	trainer->test(&measurers);
}

	

//=================== Create the MLP... =========================
ConnectedMachine *createMachine(Allocator *allocator, int n_inputs, int n_hu, int n_outputs) {
	ConnectedMachine *mlp = new(allocator) ConnectedMachine();
	if(n_hu > 0)  {
	
		//Set the first layer (input -> hidden units)
		Linear *c1 = new(allocator) Linear(n_inputs, n_hu);
		//  c1->setROption("weight decay", weight_decay);
		mlp->addFCL(c1);    
		
		//Set the second layer (threshold in hidden units)
		Tanh *c2 = new(allocator) Tanh(n_hu);
		mlp->addFCL(c2);

		//Set the last layer (Output value)
		Linear *c3 = new(allocator) Linear(n_hu, n_outputs);
		// c3->setROption("weight decay", weight_decay);
		mlp->addFCL(c3);
		}

	// Initialize the MLP
	mlp->build();
	mlp->setPartialBackprop();

	return mlp;
}


//=================== The Trainer ===============================
StochasticGradient *createTrainer(Allocator *allocator, ConnectedMachine *mlp, LearningOpt *opt) {

	printf("Trainer creation...\n");

	// The criterion for the StochasticGradient (MSE criterion)
	Criterion *criterion = NULL;
	criterion = new(allocator) MSECriterion(mlp->n_outputs);

	// The Gradient Machine Trainer
	StochasticGradient *trainer = new(allocator) StochasticGradient(mlp, criterion);

	if(opt!=NULL) {
		trainer->setIOption("max iter",opt->max_iter);
		trainer->setROption("end accuracy", opt->accuracy);
		trainer->setROption("learning rate", opt->learning_rate);
		trainer->setROption("learning rate decay", opt->decay);
	}
	return trainer;
}


