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
#include "NullXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data
#include "FileListCmdOption.h"

/* Others */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32

// warning C4291: 'void *Torch::Object::operator new(size_t,Torch::Allocator *)' : 
// no matching operator delete found; memory will not be freed if initialization throws an exception
#pragma warning( disable : 4291 )

#endif

#define STR_BUFFER_SIZE 64
#define NBR_MODEL_FILES 2

/*
 *
 */
enum MlpMode{
	ALL,
	TRAIN,
	TEST
}; 

using namespace Torch;

/*
 *
 */
struct MlpParam{
	int * n_inputs;
	//int n_outputs;
	char * files_prefix; 
	int n_classes; 
	char ** training_files; 
	char ** validation_files; 
	char ** testing_files; 
	DiagonalGMM ** gmms; 

	//model options
	int n_gaussians;

	//learning options
	real threshold;
	real prior;
	int max_iter_kmeans;
	int max_iter_gmm;
	real accuracy;
	//int k_fold;

	//misc options
	bool validation; 
	int the_seed;
	int max_load;
	char* dir_name;
	char* model_file_suffix;
	char ** model_files; 
	bool binary_mode;
	bool norm; 

	char* valid_file;
	char* suffix; 
};


/* Prototype */
void postInit(Allocator *allocator, MlpParam * param);
void clean(MlpParam * param);
void processCmd(Allocator *allocator, CmdLine * cmd, MlpParam * param); 
void initializeThreshold(DataSet* data, real* thresh, real threshold); 
void error(DiagonalGMM ** gmms, int n_classes, DataSet * data, int classe_id); 
KMeans * createKMeans(Allocator *allocator, MlpParam *param, int n); 
EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param); 
DiagonalGMM *createMachine(Allocator *allocator, DataSet* data, EMTrainer * kmeans_trainer, MeasurerList * kmeans_measurers, MlpParam *param, int n); 
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param); 

void Train(Allocator *allocator, MlpParam *param,CmdLine *cmd, int n);
//void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd);

int main(int argc, char **argv)
{
	message("Pattern Classification and Machine Learning - GMM"); 

	// Used for memory management
	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();

	// Structure supporting all parameters
	MlpParam param;
	//param.n_files = NBR_MODEL_FILES; 

	// Construct the command line
	CmdLine cmd;

	// Process command line parameters and options
	processCmd(allocator,&cmd, &param); 

	// Read the command line and retrieve the master switch option
	//MlpMode mode = (MlpMode)cmd.read(argc, argv);
	cmd.read(argc, argv);
	cmd.setWorkingDirectory(param.dir_name);
	message("WorkingDirectory:%s", param.dir_name);

	//
	postInit(allocator, &param); 
	

	for (int n=0; n < param.n_classes; n++) { 
		Train(allocator, &param, &cmd, n);
	}

	//=================== Select the mode ==========================
	//switch(mode) {
	//	case train:
	//		// train and obtain model for all of the files, each one representing a classe
	//		for (int n=0; n < param.n_classes; n++) { 
	//			message("model:%s", param.model_files[n]); 
	//			train(allocator, &param, &cmd, n);
	//		}
	//		break;
	//	case test:
	//		//testing(allocator,&param,&cmd);
	//		warning("not implemented");
	//		break;
	//	case all:
	//		warning("not implemented");
	//		break;
	//}

	//clean
	clean(&param); 

	delete allocator;
	return(0);
}


/*
 *
 */
void Train(Allocator *allocator, MlpParam *param, CmdLine *cmd, int n) {

	//================== Check if we need to validate the data ===================
	//bool validation(false);
	//if(strcmp(param->valid_file,"NULL") != 0) validation=true;
	
	message("------------- Init training [%s] -------------", param->training_files[n]);

	//=================== DataSets  ===================
	// Create a training data set
	DataSet *data_training = new(allocator) MatDataSet(param->training_files[n], -1, 0);
	param->n_inputs[n] = data_training->n_inputs; 
	message("\t=>%s loaded/n_inputs:%d", param->training_files[n], param->n_inputs[n]); 

	// Create a validation data set
	DataSet *data_validation = new(allocator) MatDataSet(param->validation_files[n], -1, 0);
	message("\t=>%s loaded/n_inputs:%d", param->validation_files[n], data_validation->n_inputs); 

	// Create a testing data set
	DataSet *data_testing = new(allocator) MatDataSet(param->testing_files[n], -1, 0);
	message("\t=>%s loaded/n_inputs:%d", param->testing_files[n], data_testing->n_inputs); 

	message("Train:mv_norm"); 
	// Normalization
	MeanVarNorm * mv_norm_training = NULL, * mv_norm_validation = NULL, * mv_norm_testing = NULL; 
	if (param->norm){
		// Computes means and standard deviation
		mv_norm_training = new(allocator) MeanVarNorm(data_training);
		mv_norm_validation = new(allocator) MeanVarNorm(data_validation);
		mv_norm_testing = new(allocator) MeanVarNorm(data_testing);

		// Normalizes the data set
		data_training->preProcess(mv_norm_training);
		data_validation->preProcess(mv_norm_validation);
		data_testing->preProcess(mv_norm_testing);
	}
	message("\t=>done");


	//=================== The Machine and its trainer  ===================	

	//create KMeans
	KMeans * kmeans = createKMeans(allocator, param, n); 

	// the kmeans measurer
	MeasurerList kmeans_measurers;

	// kmeans measurer file
	char kmeans_train_val[STR_BUFFER_SIZE]; 
	sprintf(kmeans_train_val, "gmm/kmeans_train_val_%d", n);

	//a NLL measurer
	//NLLMeasurer nll_kmeans_measurer(kmeans->log_probabilities, data_training, cmd->getXFile(kmeans_train_val) );
	NLLMeasurer nll_kmeans_measurer(kmeans->log_probabilities, data_training, new(allocator) NullXFile() );

	//add measurer to the list
	kmeans_measurers.addNode(&nll_kmeans_measurer);

	//create KMeansTrainer
	EMTrainer * kmeans_trainer = createKMeansTrainer(kmeans, allocator, param); 

	//Create GMM Machine
	param->gmms[n] = createMachine(allocator, data_training, kmeans_trainer, &kmeans_measurers, param, n);

	//Create Trainer
	EMTrainer * trainer = createTrainer(allocator, param->gmms[n], param);
	

	//=================== Mesurer  ===================
	// Measurers on the training dataset
	MeasurerList measurers;

	//gmm training measurer file
	char gmm_training_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_training_measure, "gmm/gmm_training_measure_%d", n);

	//NLL measurer for training
	NLLMeasurer nll_meas_training(param->gmms[n]->log_probabilities, data_training, cmd->getXFile(gmm_training_measure));
	measurers.addNode(&nll_meas_training);

	//gmm validation measurer files
	char gmm_validation_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_validation_measure, "gmm/gmm_validation_measure_%d", n);

	//NLL measurer for validation
	NLLMeasurer nll_meas_validation(param->gmms[n]->log_probabilities, data_validation, cmd->getXFile(gmm_validation_measure));
	measurers.addNode(&nll_meas_validation);

	//gmm testing measurer files
	char gmm_testing_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_testing_measure, "gmm/gmm_testing_measure_%d", n);

	//NLL measurer for testing
	NLLMeasurer nll_meas_testing(param->gmms[n]->log_probabilities, data_testing, cmd->getXFile(gmm_testing_measure));
	measurers.addNode(&nll_meas_testing);

    //================ Save all the data in the model =====================
	//the training is done using the data_training set as reference
	trainer->train(data_training, &measurers);


	//================ Save all the data in the model =====================
	message("Saving model"); 
	DiskXFile model_(param->model_files[n], "w");
	cmd->saveXFile(&model_);
	
	if(param->norm)
		mv_norm_training->saveXFile(&model_);

	model_.taggedWrite(&param->n_gaussians, sizeof(int), 1, "n_gaussians");
	model_.taggedWrite(&data_training->n_inputs, sizeof(int), 1, "n_inputs");
	param->gmms[n]->saveXFile(&model_);

	message("\t=>done [%s]", param->model_files[n]); 

}

/*
 *
 */
KMeans * createKMeans(Allocator *allocator, MlpParam *param, int n){

	message("createKMeans");

	// create a KMeans object to initialize the GMM
	KMeans * kmeans = new(allocator) KMeans(param->n_inputs[n], param->n_gaussians);
	kmeans->setROption("prior weights",param->prior);

	message("\t=>done");

	return kmeans; 
}

/*
 *
 */
EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param){

	message("createKMeansTrainer");

	// the kmeans trainer
	EMTrainer * kmeans_trainer = new(allocator) EMTrainer(kmeans);
	kmeans_trainer->setROption("end accuracy", param->accuracy);
	kmeans_trainer->setIOption("max iter", param->max_iter_kmeans);

	message("\t=>done");

	return kmeans_trainer; 
}


/*
 * Create the GMM Machine...
 */
DiagonalGMM *createMachine(Allocator *allocator, DataSet * data, EMTrainer * kmeans_trainer, MeasurerList * kmeans_measurers, MlpParam *param, int n) {

	message("createMachine");

	// create the GMM
	DiagonalGMM * gmm = new(allocator) DiagonalGMM(param->n_inputs[n], param->n_gaussians, kmeans_trainer);
	
	// set the training options
	real* thresh = (real*)allocator->alloc( param->n_inputs[n] * sizeof(real) );
	initializeThreshold(data, thresh, param->threshold);	
	gmm->setVarThreshold(thresh);
	gmm->setROption("prior weights",param->prior);
	gmm->setOOption("initial kmeans trainer measurers", kmeans_measurers);

	message("\t=>done");

	return gmm;
}


/*
 * The trainer
 */
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param) {

	message("createTrainer");

	// The Gradient Machine Trainer
	EMTrainer * trainer = new(allocator) EMTrainer(gmm);
	trainer->setIOption("max iter", param->max_iter_gmm);
	trainer->setROption("end accuracy", param->accuracy);

	message("\t=>done");

	return trainer;
}


//=================== Functions =============================== 

/*
 * Compute threshold
 */
void initializeThreshold(DataSet* data, real* thresh, real threshold)
{
	MeanVarNorm norm(data);
	real*	ptr = norm.inputs_stdv;
	real* p_var = thresh;
	for(int i=0;i<data->n_inputs;i++)
		*p_var++ = *ptr * *ptr++ * threshold;
}

/*
 *
 */
void processCmd( Allocator *allocator, CmdLine * cmd, MlpParam * param){

	// Construct the command line
	//FileListCmdOption file_list("file name", "the list files or one data file");
	//file_list.isArgument(true);

	message("processCmd()");

	// Put the help line at the beginning
	//cmd->info(help);

	// Train mode
	//cmd->addMasterSwitch("--train");
	cmd->addText("\nArguments:");
	cmd->addSCmdArg("file_prefix", &(param->files_prefix), "files prefix for current dataset");
	cmd->addICmdArg("nbr_classes", &(param->n_classes), "number of classes");
	//cmd->addCmdOption(&file_list);
	//cmd->addSCmdArg("file", &(param->file), "the train file");
	//cmd->addICmdArg("n_inputs", &(param->n_inputs), "input dimension of the data", true);
	//cmd->addICmdArg("n_targets", &(param->n_outputs), "output dim. (regression) or # of classes (classification)", true);

	cmd->addText("\nModel Options:");
	cmd->addICmdOption("-n_gaussians", &(param->n_gaussians), 10, "number of Gaussians");

	cmd->addText("\nLearning Options:");
	cmd->addRCmdOption("-threshold", &(param->threshold), 0.001f, "variance threshold");
	cmd->addRCmdOption("-prior", &(param->prior), 0.001f, "prior on the weights");
	cmd->addICmdOption("-iterk", &(param->max_iter_kmeans), 25, "max number of iterations of KMeans");
	cmd->addICmdOption("-iterg", &(param->max_iter_gmm), 25, "max number of iterations of GMM");
	cmd->addRCmdOption("-e", &(param->accuracy), 0.0000005f, "end accuracy");
	//cmd->addICmdOption("-kfold", &(param->k_fold), -1, "number of folds, if you want to do cross-validation");

	cmd->addText("\nMisc Options:");
	cmd->addICmdOption("-seed", &(param->the_seed), -1, "the random seed");
	cmd->addICmdOption("-load", &(param->max_load), -1, "max number of examples to load for train");
	cmd->addSCmdOption("-dir", &(param->dir_name), ".", "directory to save measures");
	cmd->addSCmdOption("-save", &(param->model_file_suffix), "gmm/gmm_model", "model file suffix");
	cmd->addBCmdOption("-bin", &(param->binary_mode), false, "binary mode for files");
	cmd->addBCmdOption("-norm", &(param->norm), true, "normalize the datas");

	//from bp.cc
	cmd->addSCmdOption("-valid", &(param->valid_file),"NULL","the valid file");
	cmd->addSCmdOption("-suffix", &(param->suffix),"","Use a suffix for MSE (train/valid) error");

	/*cmd->addMasterSwitch("--test");
	cmd->addSCmdArg("model", &(param->save_model_file), "the model file");
	cmd->addSCmdArg("file", &(param->file), "the train file");*/


	//// Retrain mode
	//cmd->addMasterSwitch("--retrain");
	//cmd->addText("\nArguments:");
	//cmd->addSCmdArg("model", &model_file, "the model file");
	//cmd->addCmdOption(&file_list);

	//cmd->addRCmdOption("-threshold", &threshold, 0.001, "variance threshold");
	//cmd->addRCmdOption("-prior", &prior, 0.001, "prior on the weights");
	//cmd->addICmdOption("-iterg", &max_iter_gmm, 25, "max number of iterations of GMM");
	//cmd->addRCmdOption("-e", &accuracy, 0.00001, "end accuracy");

	//cmd->addText("\nMisc Options:");
	//cmd->addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd->addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd->addSCmdOption("-save", &save_model_file, "", "the model file");
	//cmd->addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd->addBCmdOption("-norm", &norm, false, "normalize the datas");

	//// Test mode
	//cmd->addMasterSwitch("--test");
	//cmd->addText("\nArguments:");
	//cmd->addSCmdArg("model", &model_file, "the model file");
	//cmd->addCmdOption(&file_list);

	//cmd->addText("\nMisc Options:");
	//cmd->addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
	//cmd->addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	//cmd->addBCmdOption("-bin", &binary_mode, false, "binary mode for files");
	//cmd->addBCmdOption("-norm", &norm, false, "normalize the datas");


	//file_list
	//param->files = file_list.file_names; 
	//param->n_files = file_list.n_files; 

	//message("processCmd():param->model_files / n_files:%d", param->n_files);

	//param->model_files = new char*[param->n_files]; 
	//message("processCmd():param->model_files / sizeof:%d", param->model_files != NULL ? sizeof(param->model_files) : -1 ); 
	//message("sizeof param->model_files:%d -", sizeof(param->model_files)); 
	//for (int i=0; i < param->n_classes; i++){
	//	 param->model_files[i] = new char[STR_BUFFER_SIZE]; 
	//	 //message("processCmd():param->model_files[%d] / sizeof:%d", i, param->model_files[i] != NULL ? sizeof(param->model_files[i]) : -1 ); 
	//	 sprintf(param->model_files[i], "gmm/gmm_model_%d.dat", i); 
	//	 //message("model:%d - %s", i, param->model_files[i]); 
	//}
	message("\t=>done");
}

/*
 *
 */
void postInit(Allocator * allocator, MlpParam * param){

	param->n_inputs = new int[param->n_classes]; 
	param->training_files = new char*[param->n_classes];  
	param->validation_files = new char*[param->n_classes]; 
	param->testing_files = new char*[param->n_classes]; 
	param->model_files = new char*[param->n_classes];
	
	
	param->gmms = new DiagonalGMM*[param->n_classes]; 

	for (int i=0; i < param->n_classes; i++){

		// prepare training filenames
		param->training_files[i] = new char[STR_BUFFER_SIZE]; 
		sprintf(param->training_files[i], "%s_training_%d.data", param->files_prefix, i);

		// prepare validation filenames
		param->validation_files[i] = new char[STR_BUFFER_SIZE]; 
		sprintf(param->validation_files[i], "%s_validation_%d.data", param->files_prefix, i);

		// prepare testing filenames
		param->testing_files[i] = new char[STR_BUFFER_SIZE]; 
		sprintf(param->testing_files[i], "%s_test_%d.data", param->files_prefix, i);

		// prepare model filenames
		param->model_files[i] = new char[STR_BUFFER_SIZE]; 
		sprintf(param->model_files[i], "%s_%d.dat", param->model_file_suffix, i); 
	}

}

/*
 *
 */
void clean(MlpParam * param){

	for (int i=0; i < param->n_classes; i++){
		delete[] param->training_files[i]; 
		delete[] param->validation_files[i]; 
		delete[] param->testing_files[i];
		delete[] param->model_files[i];
	}
	delete[] param->training_files; 
	delete[] param->validation_files; 
	delete[] param->testing_files; 
	delete[] param->model_files; 
}

/*
 *
 */
void error(DiagonalGMM ** gmms, int n_classes, DataSet * data, int classe_id){

}