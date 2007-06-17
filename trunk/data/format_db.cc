The code for loading the data and setting the measurer is not relevant, so we decide to insert it only for the back-propagation in the appendix. We also use a structure Param to ease the use of function with parameters which is inside the code of back-propagation.
Part B,C,D of the appendix only contains relevant code.

A. Back-Propagation

/* A structure is used to ease the use of parameters*/
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
...
	
	//=================== Loading & Normalize Data  ===================
	MatDataSet *mat_vdata = new(allocator) MatDataSet(param->valid_file, param->n_inputs, param->n_outputs);
	MatDataSet *mat_data = new(allocator) MatDataSet(param->file, param->n_inputs, param->n_outputs);
	MeanVarNorm *mv_norm = new(allocator) MeanVarNorm(mat_data);
       	mat_data->preProcess(mv_norm);
	mat_vdata->preProcess(mv_norm);

	//Setting the class label type for ClassMeasurer
	Sequence *class_labels = new(allocator) Sequence(2,1);
	class_labels->frames[0][0] =  0;
	class_labels->frames[0][1] =  1;
	DataSet *data = new(allocator) ClassFormatDataSet(mat_data, class_labels);
	DataSet *vdata = new(allocator) ClassFormatDataSet(mat_vdata, class_labels);
    	TwoClassFormat *class_format = new(allocator) TwoClassFormat(data);

	//=================== Measurer  ===================
	char mse_train_fname[256] = "MSE_train";
	strcat(mse_train_fname,param->suffix);
	DiskXFile *mse_train_file = new(allocator) DiskXFile(mse_train_fname, "w");
	MSEMeasurer *mse_meas = new(allocator) MSEMeasurer(mlp->outputs, data, mse_train_file);
	measurers.addNode(mse_meas);	
	ClassMeasurer *class_meas = new(allocator) ClassMeasurer(mlp->outputs, data, class_format, cmd->getXFile("class_err"));
	measurers.addNode(class_meas);

...

trainer->train(data, &measurers);

B. Support Vector Machine
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
...

	//Use to format the matdatset class label from 0,1 to -1,1
	Sequence *class_labels = new(allocator) Sequence(2, 1);
	class_labels->frames[0][0] = -1;
	class_labels->frames[1][0] = 1;
	DataSet *data = new(allocator) ClassFormatDataSet(mat_data, class_labels);
...

	//We use a classmeasurer to get the classification error.
	MeasurerList measurers;
	TwoClassFormat *class_format = new(allocator) TwoClassFormat(data);
	char class_train_fname[256] = "Class_train";
	strcat(class_train_fname,param->suffix);
	DiskXFile *class_train_file = new(allocator) DiskXFile(class_train_fname, "a");
	ClassMeasurer *class_meas = new(allocator) ClassMeasurer(svm->outputs, data, class_format, class_train_file);	
	measurers.addNode(class_meas);

...
	trainer.train(data, NULL);
	message("%d SV with %d at bounds", svm->n_support_vectors, svm->n_support_vectors_bound);
	trainer.test(&measurers);



C. Gaussian Mixture Model
D. Format the database

#define NOF_PATTERNS 569
#define NOF_DIMS 30
#define NOF_CLASSES 1
#define RATIO_TRAIN 0.02
#define RATIO_VALID 0.6
#define ID_VS_OTHERS 0

using namespace std;


/* Class that is use to organize the patterns */
class Pattern {

public:
	int id;
	string code;
	string values;

	//Constructor by default
	Pattern():
		id(0), code("0"), values("")
	{}

	//Destructor 
	~Pattern() {}

	void setCode(string name) {
		if(name.compare("M") == 0) {
			id=1;
			code="1"
		}
		else {
			id=1;
			code="1"
		}
	}

	void setValues(string str) {
		for(int i(0); i<str.length(); i++) {
			if(str[i] == ',') str[i]=' ';
		}
		values = str;
	}

	friend ostream& operator<<(ostream& out, const Pattern&  p);
};



// opérateur pour l'affichage d'un pattern
ostream& operator<<(ostream& out, const Pattern&  p)
{
	out << p.values << " " << p.code;
	return out;
}
	
struct SortByCode
{
    // Object version 
    bool operator ()(const Pattern & p1,const Pattern & p2 ) const 
    {
        return p1.id < p2.id; 
    }

};

class iotaGen 
{
public:
  iotaGen (int start = 0) : current(start) { }
  int operator() () { return current++; }
private:
  int current;
};


// do it again, with explicit random number generator
struct RandomInteger {
      int operator() (int m) { return rand() % m; }
} randomize;

//Randomly shuffle a vector starting from start to end.
vector<int> randperm(int start,int end)
{
  // first make the vector containing 1 2 3 ... 10
  std::vector<int> numbers(end-start);
  std::generate(numbers.begin(), numbers.end(), iotaGen(start));

  // then randomly shuffle the elements
  std::random_shuffle(numbers.begin(), numbers.end(),randomize);

	return numbers;
 }