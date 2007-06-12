/* The measurers */
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "OneHotClassFormat.h"
#include "TwoClassFormat.h"

/* The trainer */
#include "QCTrainer.h"
#include "Random.h"

/* The Machine */
#include "SVMClassification.h"

/* Management of files */
#include "MatDataSet.h"
#include "ClassFormatDataSet.h" //????
#include "DiskXFile.h"
#include "CmdLine.h"
#include "MeanVarNorm.h" //Normalize loaded data

#define TRAIN 0
#define TEST 1

struct SvmParam{
	char *file;
	char *valid_file;
	char *suffix;
	char *model_file;

	int n_classes;
	real stdv;
	real c_cst;

	bool normalize;
	real accuracy;
	real cache_size;
	int mode;
	int iter_shrink;
};


using namespace Torch;


/* Prototype */
SVM *createMachine(Allocator *allocator, SvmParam *param); 
void TrainingKernels(Allocator *allocator, SVM **svms, MatDataSet *data, DiskXFile *model, SvmParam *param); 
void Training(Allocator *allocator, SvmParam *param,CmdLine *cmd);
//void Testing(Allocator *allocator, SvmParam *param, CmdLine *cmd);

int main(int argc, char **argv)
{

	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();
	SvmParam param;


	//=================== The command-line ==========================

	// Construct the command line
	CmdLine cmd;

	// Put the help line at the beginning
	//  cmd.info(help);

	// Train mode
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("file", &(param.file), "the train file");
	cmd.addICmdArg("n_classes", &(param.n_classes), "the number of classes", true);

	cmd.addText("\nLearning Options:"); 
	cmd.addRCmdOption("-std", &(param.stdv), 10., "the std parameter in the gaussian kernel [exp(-|x-y|^2/std^2)]", true);
	cmd.addRCmdOption("-c", &(param.c_cst), 100., "trade off cst between error/margin");
	cmd.addRCmdOption("-e", &(param.accuracy), 0.001, "end accuracy");
	cmd.addRCmdOption("-m", &(param.cache_size), 50., "cache size in Mo");
	cmd.addICmdOption("-h", &(param.iter_shrink), 100, "minimal number of iterations before shrinking");

	cmd.addText("\nMisc Options:");
	cmd.addBCmdOption("-norm", &(param.normalize),true,"Apply normalization to the database");
	cmd.addSCmdOption("-valid", &(param.valid_file),"NULL","the valid file");
	cmd.addSCmdOption("-model", &(param.model_file),"bp_model.dat","the model file");
	cmd.addSCmdOption("-suffix", &(param.suffix),"","Use a suffix for MSE (train/valid) error");

	cmd.addMasterSwitch("--test");
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("model", &(param.model_file), "the model file");
	cmd.addSCmdArg("file", &(param.file), "the test file");


	// Read the command line
	param.mode = cmd.read(argc, argv);

	printf("mode is:%d\n",param.mode);
	//=================== Select the mode ==========================
	switch(param.mode) {
		case TRAIN:
			Training(allocator,&param,&cmd);
			break;
		case TEST:
			//Testing(allocator,&param,&cmd);
			break;
	}

	delete allocator;
	return(0);
}



void Training(Allocator *allocator, SvmParam *param, CmdLine *cmd) {

	//================== Check if we need to validate the data ===================
	bool validation(false);
	if(strcmp(param->valid_file,"NULL") != 0) validation=true;

	//=================== The Model file  ===================	
	DiskXFile *model = new(allocator) DiskXFile(param->model_file, "w");
 	cmd->saveXFile(model);

	printf("Start Training ... \n");
	//=================== The Machine  ===================	
		
	Random::seed();
	//Create Machine
	SVM *svm = createMachine(allocator,param);
	
	 //=================== The Trainer ===============================

	QCTrainer trainer(svm);
	if(param->mode == 0)
	{
	trainer.setROption("end accuracy", param->accuracy);
	trainer.setIOption("iter shrink", param->iter_shrink);
	}

	//=================== DataSets  ===================
	//Create a data set
	MatDataSet *mat_data = new(allocator) MatDataSet(param->file, -1, param->n_classes);
	//Computes means and standard deviation
	if(param->normalize) {
		MeanVarNorm *mv_norm = new(allocator) MeanVarNorm(mat_data);
		//Normalizes the data set
		mat_data->preProcess(mv_norm);
		mv_norm->saveXFile(model);
	}
	
	 Sequence *class_labels = new(allocator) Sequence(2, 1);
	class_labels->frames[0][0] = -1;
	class_labels->frames[1][0] = 1;
	DataSet *data = new(allocator) ClassFormatDataSet(mat_data, class_labels);

	//=================== Measurer ===================
	// The list of measurers
	MeasurerList measurers;
/*
	TwoClassFormat *class_format = new(allocator) TwoClassFormat(data);
      ClassMeasurer *class_meas = new(allocator) ClassMeasurer(svm->outputs, data, class_format, cmd->getXFile("the_class_err"));
      measurers.addNode(class_meas);
     */ 
      
	// The mean square error file on disk
	/*char mse_train_fname[256] = "MSE_train";
	strcat(mse_train_fname,param->suffix);
	DiskXFile *mse_train_file = new(allocator) DiskXFile(mse_train_fname, "w");
	MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, mse_train_file);
	measurers.addNode(mse_meas);
	*/

	/*
	OneHotClassFormat *class_format = new(allocator) OneHotClassFormat(param->n_classes);
	char class_train_fname[256] = "Class_train";
	strcat(class_train_fname,param->suffix);

    	ClassMeasurer *class_meas = new(allocator) ClassMeasurer(svms->outputs, data, class_format, cmd->getXFile(class_train_fname));
   	measurers.addNode(class_meas);
	*/

	//================= Validation (Data & Mesurer) ==============
	if(validation) {
		/*
		printf("Load Validation data...\n");
		char mse_valid_fname[256] = "MSE_valid";
		strcat(mse_valid_fname,param->suffix);
		DataSet *vdata =  new(allocator) MatDataSet(param->valid_file, param->n_inputs, param->n_outputs);
		vdata->preProcess(mv_norm);
		DiskXFile *mse_valid_file = new(allocator) DiskXFile(mse_valid_fname, "w");
		MSEMeasurer *mse_valid_meas = new(allocator) MSEMeasurer(mlp->outputs, vdata, mse_valid_file);
		measurers.addNode(mse_valid_meas);

		char class_valid_fname[256] = "Class_valid";
		strcat(class_valid_fname,param->suffix);

        	ClassMeasurer *valid_class_meas = new(allocator) ClassMeasurer(mlp->outputs, vdata, class_format, cmd->getXFile(class_valid_fname));
       		measurers.addNode(valid_class_meas);*/
	}

	//================ Train each kernels =====================
      	  trainer.train(data, NULL);
    message("%d SV with %d at bounds", svm->n_support_vectors, svm->n_support_vectors_bound);
    
    /*
    DiskXFile model_(model_file, "w");
    cmd.saveXFile(&model_);
    if(normalize)
      mv_norm->saveXFile(&model_);
    svm->saveXFile(&model_);
*/
	//================ Save all the data in the model =====================
	//mlp->saveXFile(&model_);
}

/*
void Testing(Allocator *allocator, MlpParam *param, CmdLine *cmd) {

	//=================== Load Model =====================
	DiskXFile *model = new(allocator) DiskXFile(param->model_file, "r");
	cmd->loadXFile(model);


	//=================== Validation DataSets  ===================	
	printf("Load data...\n");
	printf("%s, %d, %d, %d\n",param->file,param->n_inputs,param->n_hu,param->n_outputs);
	DataSet *data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);

	//=================== Normalize Data  ===================
	printf("Normalize data ...\n");
	MeanVarNorm *mv_norm = NULL;
	mv_norm = new(allocator) MeanVarNorm(data);
	mv_norm->loadXFile(model); //Loading value from model to know how we normalize last time
        data->preProcess(mv_norm);

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
	MSEMeasurer *test_mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, cmd->getXFile("testing_mse_err"));
        measurers.addNode(test_mse_meas);

	//=================== Test the machine ===============================
	printf("Test our built model\n");
	trainer->test(&measurers);
}*/
	

//=================== Create the MLP... =========================
SVM * createMachine(Allocator *allocator, SvmParam *param) {

	//Create the kernel type
	Kernel *kernel = new(allocator) GaussianKernel(1./(param->stdv*param->stdv));

	//Create SVM
 	SVM *svm = new(allocator) SVMClassification(kernel);
	
	
	if(param->mode == TRAIN)   {
		svm->setROption("C", param->c_cst);
		svm->setROption("cache size", param->cache_size);
	}

	return svm;
}





