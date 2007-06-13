/* The measurer */
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "NLLMeasurer.h"

/* The trainer */
#include "EMTrainer.h"
#include "MSECriterion.h" //Optimize mean square error

/* The GMM tools */
#include "KMeans.h"
#include "DiagonalGMM.h"
#include "KFold.h"

/* Management of files */
#include "MatDataSet.h"
#include "DiskXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data

#ifdef WIN32

// warning C4291: 'void *Torch::Object::operator new(size_t,Torch::Allocator *)' : 
// no matching operator delete found; memory will not be freed if initialization throws an exception
#pragma warning( disable : 4291 )

#endif

enum MlpMode{
	ALL,
	TRAIN,
	TEST
}; 

struct MlpParam{
	int n_inputs;
	int n_outputs;
	char* file; 

	//model options
	int n_gaussians;

	//learning options
	real threshold;
	real prior;
	int max_iter_kmeans;
	int max_iter_gmm;
	real accuracy;
	int k_fold;

	//misc options
	int the_seed;
	int max_load;
	char* dir_name;
	char* model_file;
	bool binary_mode;
	bool norm; 

	char* valid_file;
	char* suffix; 
};


using namespace Torch;


/* Prototype */
void initializeThreshold(DataSet* data, real* thresh, real threshold); 
KMeans * createKMeans(Allocator *allocator, MlpParam *param); 
EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param); 
DiagonalGMM *createMachine(Allocator *allocator, DataSet* data, EMTrainer * kmeans_trainer, MlpParam *param); 
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param); 

void Training(Allocator *allocator, MlpParam *param,CmdLine *cmd);
//void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd);

int main(int argc, char **argv)
{

	// used for memory management
	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();

	// structure supporting all parameters
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

	cmd.addText("\nModel Options:");
	cmd.addICmdOption("-n_gaussians", &(param.n_gaussians), 10, "number of Gaussians");

	cmd.addText("\nLearning Options:");
	cmd.addRCmdOption("-threshold", &(param.threshold), 0.001f, "variance threshold");
	cmd.addRCmdOption("-prior", &(param.prior), 0.001f, "prior on the weights");
	cmd.addICmdOption("-iterk", &(param.max_iter_kmeans), 25, "max number of iterations of KMeans");
	cmd.addICmdOption("-iterg", &(param.max_iter_gmm), 25, "max number of iterations of GMM");
	cmd.addRCmdOption("-e", &(param.accuracy), 0.0000005f, "end accuracy");
	cmd.addICmdOption("-kfold", &(param.k_fold), -1, "number of folds, if you want to do cross-validation");

	cmd.addText("\nMisc Options:");
	cmd.addICmdOption("-seed", &(param.the_seed), -1, "the random seed");
	cmd.addICmdOption("-load", &(param.max_load), -1, "max number of examples to load for train");
	cmd.addSCmdOption("-dir", &(param.dir_name), ".", "directory to save measures");
	cmd.addSCmdOption("-save", &(param.model_file), "gmm_model.dat", "the model file");
	cmd.addBCmdOption("-bin", &(param.binary_mode), false, "binary mode for files");
	cmd.addBCmdOption("-norm", &(param.norm), true, "normalize the datas");

	//from bp.cc
	cmd.addSCmdOption("-valid", &(param.valid_file),"NULL","the valid file");
	cmd.addSCmdOption("-suffix", &(param.suffix),"","Use a suffix for MSE (train/valid) error");

	/*cmd.addMasterSwitch("--test");
	cmd.addSCmdArg("model", &(param.save_model_file), "the model file");
	cmd.addSCmdArg("file", &(param.file), "the train file");*/


	//// Retrain mode
	//cmd.addMasterSwitch("--retrain");
	//cmd.addText("\nArguments:");
	//cmd.addSCmdArg("model", &model_file, "the model file");
	//cmd.addCmdOption(&file_list);

	//cmd.addRCmdOption("-threshold", &threshold, 0.001, "variance threshold");
	//cmd.addRCmdOption("-prior", &prior, 0.001, "prior on the weights");
	//cmd.addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	//cmd.addRCmdOption("-e", &accuracy, 0.00001, "end accuracy");

	//cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addSCmdOption("-save", &save_model_file, "", "the model file");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-norm", &norm, false, "normalize the datas");

	//// Test mode
	//cmd.addMasterSwitch("--test");
	//cmd.addText("\nArguments:");
	//cmd.addSCmdArg("model", &model_file, "the model file");
	//cmd.addCmdOption(&file_list);

	//cmd.addText("\nMisc Options:");
	//cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd.addBCmdOption("-norm", &norm, false, "normalize the datas");


	// Read the command line
	MlpMode mode = (MlpMode)cmd.read(argc, argv);

	//=================== Select the mode ==========================
	switch(mode) {
		case TRAIN:
			Training(allocator,&param,&cmd);
			break;
		case TEST:
			//Testing(allocator,&param,&cmd);
			warning("Not Implemented");
			break;
		case ALL:
			warning("Not Implemented");
			break;
	}

	delete allocator;
	return(0);
}



void Training(Allocator *allocator, MlpParam *param, CmdLine *cmd) {

	//================== Check if we need to validate the data ===================
	//bool validation(false);
	//if(strcmp(param->valid_file,"NULL") != 0) validation=true;
	
	message("Start Training ... ");

	//=================== DataSets  ===================
	// Create a data set
	DataSet *data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);

	// Normalization
	if (param->norm){
		// Computes means and standard deviation
		MeanVarNorm *mv_norm = new(allocator) MeanVarNorm(data);

		// Normalizes the data set
		data->preProcess(mv_norm);
	}


	//=================== The Machine and its trainer  ===================	

	//create KMeans
	KMeans * kmeans = createKMeans(allocator, param); 

	//create KMeansTrainer
	EMTrainer * kmeans_trainer = createKMeansTrainer(kmeans, allocator, param); 

	//Create GMM Machine
	DiagonalGMM * gmm = createMachine(allocator, data, kmeans_trainer, param);

	//Create Trainer
	EMTrainer * trainer = createTrainer(allocator, gmm, param);
	

	//=================== Mesurer  ===================
	// Measurers on the training dataset
	MeasurerList measurers;

	NLLMeasurer nll_meas(gmm->log_probabilities, data, cmd->getXFile("gmm_train_val"));
	measurers.addNode(&nll_meas);

	//// The mean square error file on disk
	//char mse_train_fname[256] = "MSE_train";
	//strcat(mse_train_fname,param->suffix);
	//DiskXFile *mse_train_file = new(allocator) DiskXFile(mse_train_fname, "w");
	//MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, mse_train_file);
	//measurers.addNode(mse_meas);


	//================= Train the MLP and find the weight ====================
    trainer->train(data, &measurers);

	
    //================ Save all the data in the model =====================
	/*DiskXFile model_(param->model_file, "w");
	cmd->saveXFile(&model_);
	mv_norm->saveXFile(&model_);
	mlp->saveXFile(&model_);*/
}

//void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd) {
//
//	//=================== Load Model =====================
//	DiskXFile *model = new(allocator) DiskXFile(param->model_file, "r");
//	cmd->loadXFile(model);
//
//
//	//=================== Validation DataSets  ===================	
//	printf("Load data...\n");
//	printf("%s, %d, %d, %d\n",param->file,param->n_inputs,param->n_hu,param->n_outputs);
//	DataSet *data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);
//
//	//=================== Normalize Data  ===================
//	printf("Normalize data ...\n");
//	MeanVarNorm *mv_norm = NULL;
//	mv_norm = new(allocator) MeanVarNorm(data);
//	mv_norm->loadXFile(model); //Loading value from model to know how we normalize last time
//        data->preProcess(mv_norm);
//
//	//=================== The Machine and its trainer  ===================	
//	printf("Building Machine...\n");
//	//Create Machine structure
//	ConnectedMachine *mlp = createMachine(allocator,param);
//	mlp->loadXFile(model); //Loading MLP weight structure
//	//Create trainer structure
//	StochasticGradient *trainer = createTrainer(allocator, mlp);
//
//	//=================== Mesurer  ===================
//	printf("Building Measurer...\n");
//	// The list of measurers
//	MeasurerList measurers;
//
//	// Measurers on the training dataset
//	MSEMeasurer *test_mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, cmd->getXFile("testing_mse_err"));
//        measurers.addNode(test_mse_meas);
//
//	//=================== Test the machine ===============================
//	printf("Test our built model\n");
//	trainer->test(&measurers);
//}

KMeans * createKMeans(Allocator *allocator, MlpParam *param){

	// create a KMeans object to initialize the GMM
	KMeans * kmeans = new(allocator) KMeans(param->n_inputs, param->n_gaussians);
	kmeans->setROption("prior weights",param->prior);
}

EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param){

	// the kmeans trainer
	EMTrainer * kmeans_trainer = new(allocator) EMTrainer(kmeans);
	kmeans_trainer->setROption("end accuracy", param->accuracy);
	kmeans_trainer->setIOption("max iter", param->max_iter_kmeans);

	return kmeans_trainer; 
}


//=================== Create the GMM Machine... =========================
DiagonalGMM *createMachine(Allocator *allocator, DataSet * data, EMTrainer * kmeans_trainer, MlpParam *param) {


	// create the GMM
	DiagonalGMM * gmm = new(allocator) DiagonalGMM(param->n_inputs, param->n_gaussians, kmeans_trainer);
	
	// set the training options
	real* thresh = (real*)allocator->alloc( param->n_inputs * sizeof(real) );
	initializeThreshold(data, thresh, param->threshold);	
	gmm->setVarThreshold(thresh);
	gmm->setROption("prior weights",param->prior);
	//gmm->setOOption("initial kmeans trainer measurers", &kmeans_measurers);

	return gmm;
}


//=================== The Trainer ===============================
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param) {

	message("Trainer creation...");

	// The Gradient Machine Trainer
	EMTrainer * trainer = new(allocator) EMTrainer(gmm);
	trainer->setIOption("max iter", param->max_iter_gmm);
	trainer->setROption("end accuracy", param->accuracy);

	return trainer;
}


//=================== Functions =============================== 

void initializeThreshold(DataSet* data, real* thresh, real threshold)
{
	MeanVarNorm norm(data);
	real*	ptr = norm.inputs_stdv;
	real* p_var = thresh;
	for(int i=0;i<data->n_inputs;i++)
		*p_var++ = *ptr * *ptr++ * threshold;
}