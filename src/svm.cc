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
void Training(Allocator *allocator, SvmParam *param,CmdLine *cmd);

int main(int argc, char **argv)
{

	SvmParam param;
	int ite_start,ite_end,ite_step;
	char *dir_name;
	
	Allocator *allocator = new Allocator;
	DiskXFile::setLittleEndianMode();

	//=================== The command-line ==========================

	// Construct the command line
	CmdLine cmd;

	// Put the help line at the beginning
	//  cmd.info(help);

	// Train mode
	cmd.addText("\nArguments:");
	cmd.addSCmdArg("file", &(param.file), "the train file");

	cmd.addText("\nLearning Options:"); 
	cmd.addICmdOption("n_classes", &(param.n_classes),1, "the number of classes", true);
	cmd.addRCmdOption("-std", &(param.stdv), 10., "the std parameter in the gaussian kernel [exp(-|x-y|^2/std^2)]", true);
	cmd.addRCmdOption("-c", &(param.c_cst), 100., "trade off cst between error/margin");
	cmd.addRCmdOption("-e", &(param.accuracy), 0.01, "end accuracy");
	cmd.addRCmdOption("-m", &(param.cache_size), 50., "cache size in Mo");
	cmd.addICmdOption("-h", &(param.iter_shrink), 100, "minimal number of iterations before shrinking");

	cmd.addText("\nMisc Options:");
	cmd.addBCmdOption("-norm", &(param.normalize),false,"Apply normalization to the database");
	cmd.addSCmdOption("-valid", &(param.valid_file),"NULL","the valid file");
	cmd.addSCmdOption("-model", &(param.model_file),"bp_model.dat","the model file");

	cmd.addText("\nIteration Options:");
	cmd.addSCmdOption("-suffix", &(param.suffix),"","Use a suffix for MSE (train/valid) error");
	cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
	cmd.addICmdOption("-ite_start", &ite_start, 20, "starting std iteration");
	cmd.addICmdOption("-ite_step", &ite_step, 1, "step of std iteration n");
	cmd.addICmdOption("-ite_end", &ite_end, 100, "end std iteration");
	


	// Read the command line
	param.mode = cmd.read(argc, argv);

	printf("Debut \n");
	fflush(stdout);
	printf("mode is:%d, %s\n",param.mode,dir_name);
	//=================== Select the mode ==========================
	switch(param.mode) {
		case TRAIN:
			for(int i=ite_start;i<ite_end;i+=ite_step) {
				param.stdv=i;
				Training(allocator,&param,&cmd);
			}
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
	trainer.setROption("end accuracy", param->accuracy);
	trainer.setIOption("iter shrink", param->iter_shrink);

	//=================== DataSets  ===================
	//Create a data set
	MatDataSet *mat_data = new(allocator) MatDataSet(param->file, -1, param->n_classes);
	//Computes means and standard deviation
	MeanVarNorm *mv_norm = NULL;
	if(param->normalize) {
		mv_norm = new(allocator) MeanVarNorm(mat_data);
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

	TwoClassFormat *class_format = new(allocator) TwoClassFormat(data);
	char class_train_fname[256] = "Class_train";
	strcat(class_train_fname,param->suffix);
	DiskXFile *class_train_file = new(allocator) DiskXFile(class_train_fname, "a");

	ClassMeasurer *class_meas = new(allocator) ClassMeasurer(svm->outputs, data, class_format, class_train_file);	
	measurers.addNode(class_meas);
      
	//================= Validation (Data & Mesurer) ==============
	if(validation) {
		printf("Load Validation data...\n");
		MatDataSet *mat_vdata =  new(allocator) MatDataSet(param->valid_file, -1, 1);
		if(param->normalize) mat_vdata->preProcess(mv_norm);
		DataSet *vdata = new(allocator) ClassFormatDataSet(mat_vdata, class_labels);

/*
		char mse_valid_fname[256] = "MSE_valid";
		strcat(mse_valid_fname,param->suffix);
		DiskXFile *mse_valid_file = new(allocator) DiskXFile(mse_valid_fname, "a");
		MSEMeasurer *mse_valid_meas = new(allocator) MSEMeasurer(mlp->outputs, vdata, mse_valid_file);
		measurers.addNode(mse_valid_meas);
*/
		TwoClassFormat *class_format_valid = new(allocator) TwoClassFormat(vdata);
		char class_valid_fname[256] = "Class_valid";
		strcat(class_valid_fname,param->suffix);
		DiskXFile *class_valid_file = new(allocator) DiskXFile(class_valid_fname, "a");
        	ClassMeasurer *valid_class_meas = new(allocator) ClassMeasurer(svm->outputs, vdata, class_format_valid, class_valid_file);
       		measurers.addNode(valid_class_meas);
	}

	//================ Train each kernels =====================
	trainer.train(data, NULL);
	message("%d SV with %d at bounds", svm->n_support_vectors, svm->n_support_vectors_bound);
   
	trainer.test(&measurers);
 
/*
	//================ Save all the data in the model =====================
	DiskXFile model_(model_file, "w");
	cmd.saveXFile(&model_);
	if(normalize)
	mv_norm->saveXFile(&model_);
	svm->saveXFile(&model_);
*/
}


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

