/* The measurer */
#include "NLLMeasurer.h"

/* The trainer */
#include "EMTrainer.h"

/* The GMM tools */
#include "KMeans.h"
#include "DiagonalGMM.h"

/* Management of files */
#include "MatDataSet.h"
#include "DiskXFile.h"
#include "NullXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data

/* Others */
#include <stdio.h>
#include <stdlib.h>


#ifdef WIN32

// warning C4291: 'void *Torch::Object::operator new(size_t,Torch::Allocator *)' : 
// no matching operator delete found; memory will not be freed if initialization throws an exception
#pragma warning( disable : 4291 )

#endif

#define STR_BUFFER_SIZE 64

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
	MlpParam():mv_norm(NULL){
	}; 

	int * n_inputs;
	char * files_prefix; 
	int n_classes; 
	char ** training_files; 
	char ** validation_files; 
	char ** testing_files; 

	//
	DiagonalGMM ** gmms; 
	MeanVarNorm * mv_norm; 

	char * training_suffix; 
	char * testing_suffix; 
	char * validation_suffix; 

	DataSet ** data_training; 
	DataSet ** data_testing; 
	DataSet ** data_validation; 

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
	char * result_dir; 
	bool verbose; 

	char* valid_file;
	char* prefix; 
};


/* Prototype */
void postInit(Allocator *allocator, MlpParam * param);
void loadData(Allocator *allocator, MlpParam *param, int n);
void clean(MlpParam * param);
void processCmd(Allocator *allocator, CmdLine * cmd, MlpParam * param); 
void initializeThreshold(DataSet* data, real* thresh, real threshold); 
real error(DiagonalGMM ** gmms, int n_classes, DataSet * data, int classe_id); 
KMeans * createKMeans(Allocator *allocator, MlpParam *param, int n); 
EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param); 
DiagonalGMM *createMachine(Allocator *allocator, DataSet* data, EMTrainer * kmeans_trainer, MeasurerList * kmeans_measurers, MlpParam *param, int n); 
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param); 

void Train(Allocator *allocator, MlpParam *param,CmdLine *cmd, int n);

int main(int argc, char **argv)
{
	message("Pattern Classification and Machine Learning - GMM"); 

	// Used for memory management
	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();

	// Structure supporting all parameters
	MlpParam param;

	// Construct the command line
	CmdLine cmd;

	// Process command line parameters and options
	processCmd(allocator,&cmd, &param); 

	// Read the command line and retrieve the master switch option
	//MlpMode mode = (MlpMode)cmd.read(argc, argv);
	cmd.read(argc, argv);
	cmd.setWorkingDirectory(param.dir_name);
	if(param.verbose) message("WorkingDirectory:%s", param.dir_name);

	//
	postInit(allocator, &param); 

	for (int n=0; n < param.n_classes; n++) { 
		loadData(allocator, &param, n); 
		Train(allocator, &param, &cmd, n);
	}

	message("Compute classification error");
	char error_file[STR_BUFFER_SIZE]; 
	char buffer[STR_BUFFER_SIZE]; 
	DiskXFile * errorXfile = NULL; 
	//loop through each classes
	for (int i=0; i < param.n_classes; i++){
		//compute error for each dataset
		real e_training = error(param.gmms, param.n_classes, param.data_training[i], i); 
		real e_testing = error(param.gmms, param.n_classes, param.data_testing[i], i); 
		real e_validation = error(param.gmms, param.n_classes, param.data_validation[i], i); 

		message("\tclasse(%d) / error rate training: %f, testing: %f, validation: %f", i, e_training, e_testing, e_validation); 

		//write training error
		sprintf(error_file, "gmm/gmm%s_error_%d", param.training_suffix, i);
		sprintf(buffer, "%d %f\n", param.n_gaussians, e_training); 
		errorXfile = new(allocator) DiskXFile(error_file, "a");
		errorXfile->write(buffer, (int)strlen(buffer), 1); 

		//write validation error
		sprintf(error_file, "gmm/gmm%s_error_%d", param.validation_suffix, i);
		sprintf(buffer, "%d %f\n", param.n_gaussians, e_validation); 
		errorXfile = new(allocator) DiskXFile(error_file, "a");
		errorXfile->write(buffer, (int)strlen(buffer), 1); 

		//write testing error
		sprintf(error_file, "gmm/gmm%s_error_%d", param.testing_suffix, i);
		sprintf(buffer, "%d %f\n", param.n_gaussians, e_testing); 
		errorXfile = new(allocator) DiskXFile(error_file, "a");
		errorXfile->write(buffer, (int)strlen(buffer), 1); 
	}
	if(param.verbose) message("\t=>done"); 


	//clean
	clean(&param); 

	delete allocator;
	return(0);
}

void loadData(Allocator *allocator, MlpParam *param, int n){
	//=================== DataSets  ===================
	if(param->verbose) message("Load data"); 
	// Create a training data set
	param->data_training[n] = new(allocator) MatDataSet(param->training_files[n], -1, 0);
	param->n_inputs[n] = param->data_training[n]->n_inputs; 
	if(param->verbose) message("\t=>%s loaded/n_inputs:%d", param->training_files[n], param->n_inputs[n]); 

	// Create a validation data set
	param->data_validation[n] = new(allocator) MatDataSet(param->validation_files[n], -1, 0);
	if(param->verbose) message("\t=>%s loaded/n_inputs:%d", param->validation_files[n], param->data_validation[n]->n_inputs); 

	// Create a testing data set
	param->data_testing[n] = new(allocator) MatDataSet(param->testing_files[n], -1, 0);
	if(param->verbose) message("\t=>%s loaded/n_inputs:%d", param->testing_files[n], param->data_testing[n]->n_inputs); 

	if(param->verbose) message("\t=>done");

	if(param->verbose) message("Compute mv_norm"); 
	// Normalization
	if (param->norm){
		// Computes means and standard deviation
		if (param->mv_norm == NULL) param->mv_norm = new(allocator) MeanVarNorm(param->data_training[n]);

		// Normalizes the data set
		param->data_training[n]->preProcess(param->mv_norm);
		param->data_validation[n]->preProcess(param->mv_norm);
		param->data_testing[n]->preProcess(param->mv_norm);
	}
	if(param->verbose) message("\t=>done");
}


/*
 *
 */
void Train(Allocator *allocator, MlpParam *param, CmdLine *cmd, int n) {
	
	if(param->verbose) message("------------- Init training [%s] -------------", param->training_files[n]);


	//=================== The Machine and its trainer  ===================	

	//create KMeans
	KMeans * kmeans = createKMeans(allocator, param, n); 

	// the kmeans measurer
	MeasurerList kmeans_measurers;

	// kmeans measurer file
	char kmeans_training_measure[STR_BUFFER_SIZE]; 
	sprintf(kmeans_training_measure, "%s/kmeans%s%s_measure_%d",  param->result_dir, param->prefix, param->training_suffix, n);

	//a NLL measurer
	//NLLMeasurer nll_kmeans_measurer(kmeans->log_probabilities, param->data_training[n], cmd->getXFile(kmeans_training_measure) );
	NLLMeasurer nll_kmeans_measurer(kmeans->log_probabilities, param->data_training[n], new(allocator) NullXFile() );

	//add measurer to the list
	kmeans_measurers.addNode(&nll_kmeans_measurer);

	//create KMeansTrainer
	EMTrainer * kmeans_trainer = createKMeansTrainer(kmeans, allocator, param); 

	//Create GMM Machine
	param->gmms[n] = createMachine(allocator, param->data_training[n], kmeans_trainer, &kmeans_measurers, param, n);

	//Create Trainer
	EMTrainer * trainer = createTrainer(allocator, param->gmms[n], param);
	

	//=================== Mesurer  ===================
	// Measurers on the training dataset
	MeasurerList measurers;

	//gmm training measurer file
	char gmm_training_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_training_measure, "%s/gmm%s%s_measure_%d",  param->result_dir, param->prefix, param->training_suffix, n);

	//NLL measurer for training
	NLLMeasurer nll_meas_training(param->gmms[n]->log_probabilities, param->data_training[n], cmd->getXFile(gmm_training_measure));
	measurers.addNode(&nll_meas_training);


	//message("average_examples:%s", nll_meas_training.average_examples ? "true" : "false" ); 
	//message("average_frames:%s", nll_meas_training.average_frames ? "true" : "false" ); 

	//gmm validation measurer files
	char gmm_validation_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_validation_measure, "%s/gmm%s%s_measure_%d", param->result_dir, param->prefix, param->validation_suffix, n);

	//NLL measurer for validation
	NLLMeasurer nll_meas_validation(param->gmms[n]->log_probabilities, param->data_validation[n], cmd->getXFile(gmm_validation_measure));
	measurers.addNode(&nll_meas_validation);

	//gmm testing measurer files
	char gmm_testing_measure[STR_BUFFER_SIZE]; 
	sprintf(gmm_testing_measure, "%s/gmm%s%s_measure_%d", param->result_dir, param->prefix, param->testing_suffix, n);

	//NLL measurer for testing
	NLLMeasurer nll_meas_testing(param->gmms[n]->log_probabilities, param->data_testing[n], cmd->getXFile(gmm_testing_measure));
	measurers.addNode(&nll_meas_testing);

    //================ Save all the data in the model =====================
	//the training is done using the data_training set as reference
	trainer->train(param->data_training[n], &measurers);

	//display GMM properties
	//param->gmms[n]->display();

	//================ Save all the data in the model =====================
	if(param->verbose)message("Saving model"); 
	DiskXFile model_(param->model_files[n], "w");
	cmd->saveXFile(&model_);
	
	if(param->norm)
		param->mv_norm->saveXFile(&model_);

	model_.taggedWrite(&param->n_gaussians, sizeof(int), 1, "n_gaussians");
	model_.taggedWrite(&param->data_training[n]->n_inputs, sizeof(int), 1, "n_inputs");
	param->gmms[n]->saveXFile(&model_);

	if(param->verbose) message("\t=>done [%s]", param->model_files[n]); 
}

/*
 *
 */
KMeans * createKMeans(Allocator *allocator, MlpParam *param, int n){

	if(param->verbose) message("createKMeans");

	// create a KMeans object to initialize the GMM
	KMeans * kmeans = new(allocator) KMeans(param->n_inputs[n], param->n_gaussians);
	kmeans->setROption("prior weights",param->prior);

	if(param->verbose) message("\t=>done");

	return kmeans; 
}

/*
 *
 */
EMTrainer * createKMeansTrainer(KMeans * kmeans, Allocator *allocator, MlpParam *param){

	if(param->verbose) message("createKMeansTrainer");

	// the kmeans trainer
	EMTrainer * kmeans_trainer = new(allocator) EMTrainer(kmeans);
	kmeans_trainer->setROption("end accuracy", param->accuracy);
	kmeans_trainer->setIOption("max iter", param->max_iter_kmeans);

	if(param->verbose) message("\t=>done");

	return kmeans_trainer; 
}


/*
 * Create the GMM Machine...
 */
DiagonalGMM *createMachine(Allocator *allocator, DataSet * data, EMTrainer * kmeans_trainer, MeasurerList * kmeans_measurers, MlpParam *param, int n) {

	if(param->verbose) message("createMachine");

	// create the GMM
	DiagonalGMM * gmm = new(allocator) DiagonalGMM(param->n_inputs[n], param->n_gaussians, kmeans_trainer);
	
	// set the training options
	real* thresh = (real*)allocator->alloc( param->n_inputs[n] * sizeof(real) );
	initializeThreshold(data, thresh, param->threshold);	
	gmm->setVarThreshold(thresh);
	gmm->setROption("prior weights",param->prior);
	gmm->setOOption("initial kmeans trainer measurers", kmeans_measurers);

	if(param->verbose) message("\t=>done");

	return gmm;
}


/*
 * The trainer
 */
EMTrainer *createTrainer(Allocator *allocator, DiagonalGMM *gmm, MlpParam *param) {

	if(param->verbose) message("createTrainer");

	// The Gradient Machine Trainer
	EMTrainer * trainer = new(allocator) EMTrainer(gmm);
	trainer->setIOption("max iter", param->max_iter_gmm);
	trainer->setROption("end accuracy", param->accuracy);

	if(param->verbose) message("\t=>done");

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

	if(param->verbose) message("processCmd()");

	// Put the help line at the beginning
	//cmd->info(help);

	// Train mode
	//cmd->addMasterSwitch("--train");
	cmd->addText("\nArguments:");
	cmd->addSCmdArg("file_prefix", &(param->files_prefix), "files prefix for current dataset");
	cmd->addICmdArg("nbr_classes", &(param->n_classes), "number of classes");

	cmd->addText("\nModel Options:");
	cmd->addICmdOption("-n_gaussians", &(param->n_gaussians), 10, "number of Gaussians");

	cmd->addText("\nLearning Options:");
	cmd->addRCmdOption("-threshold", &(param->threshold), 0.001f, "variance threshold");
	cmd->addRCmdOption("-prior", &(param->prior), 0.001f, "prior on the weights");
	cmd->addICmdOption("-iterk", &(param->max_iter_kmeans), 25, "max number of iterations of KMeans");
	cmd->addICmdOption("-iterg", &(param->max_iter_gmm), 25, "max number of iterations of GMM");
	cmd->addRCmdOption("-e", &(param->accuracy), 0.0000005f, "end accuracy");

	cmd->addText("\nMisc Options:");
	cmd->addICmdOption("-seed", &(param->the_seed), -1, "the random seed");
	cmd->addICmdOption("-load", &(param->max_load), -1, "max number of examples to load for train");
	cmd->addSCmdOption("-dir", &(param->dir_name), ".", "directory to save measures");
	cmd->addSCmdOption("-save", &(param->model_file_suffix), "_model", "model file suffix");
	cmd->addBCmdOption("-bin", &(param->binary_mode), false, "binary mode for files");
	cmd->addBCmdOption("-norm", &(param->norm), true, "normalize the datas");
	cmd->addSCmdOption("-training_suffix", &(param->training_suffix),"_training","Training suffix");
	cmd->addSCmdOption("-validation_suffix", &(param->validation_suffix),"_validation","Validation suffix");
	cmd->addSCmdOption("-testing_suffix", &(param->testing_suffix),"_testing","Testing suffix");
	cmd->addSCmdOption("-result_dir", &(param->result_dir),"gmm", "Result directory");
	cmd->addSCmdOption("-prefix", &(param->prefix),"0_", "File prefix");
	cmd->addBCmdOption("-verbose", &(param->verbose),false, "Display more informationss");

	if(param->verbose) message("\t=>done");
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

	param->data_training = new DataSet*[param->n_classes]; 
	param->data_testing = new DataSet*[param->n_classes]; 
	param->data_validation = new DataSet*[param->n_classes];  

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
		sprintf(param->model_files[i], "%s/gmm%s%s_%d.dat",  param->result_dir, param->prefix, param->model_file_suffix, i); 
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

	delete[] param->gmms ; 

	delete[] param->data_training; 
	delete[] param->data_testing; 
	delete[] param->data_validation; 
}

/*
 *
 */
real error(DiagonalGMM ** gmms, int n_classes, DataSet * data, int classe_id){

	real * gmm_outputs = new real[n_classes];
	//real * gmm_outputs_logp = new real[n_classes];

	int misclassified = 0; 

	//loop though all examples of the data set
	for(int i= 0; i < data->n_examples; i++){
		
		//select example 'i'
		data->setExample(i); 


		for (int j=0; j < n_classes; j++){
			//update output using EM
			gmms[j]->forward(data->inputs); 
			//gmms[j]->eMForward(data->inputs);

			//read output
			gmm_outputs[j] = (real)gmms[j]->outputs->frames[0][0]; 
			//message("[%d][%d]:%f", i, j, gmm_outputs[j]); 
		}
		//compare gmm_outputs
		int greater = 0; 
		for (int k=1; k<n_classes; k++){
			if ( gmm_outputs[k] > gmm_outputs[greater]) greater = k; 
		}

		//check classification
		if (classe_id != greater) misclassified++; 
	}

	//message("\t=>misclassified:%d, n_examples:%d", misclassified, data->n_examples); 

	return ((real)misclassified / (real)data->n_examples)*100; 
}