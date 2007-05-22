#include <iostream>
#include <fstream>
#include <string>
#include <vector>


#define NOF_PATTERNS 2310
#define NOF_DIMS 19
#define NOF_CLASSES 7
#define RATIO_TRAIN 0.2
#define RATIO_VALID 0.4

using namespace std;
class Pattern {

public:
	int id;
	string code;
	string values;

	//Constructor by default
	Pattern():
		id(0), code("0 0 0 0 0 0 0"), values("140.0 125.0 9 0.0 0.0 0.2777779 0.06296301 0.66666675 0.31111118 6.185185 7.3333335 7.6666665 3.5555556 3.4444444 4.4444447 -7.888889 7.7777777 0.5456349 -1.1218182")
	{}

	//Destructor 
	~Pattern() {}

	void setCode(string name) {
		if(name.compare("BRICKFACE") == 0) id=1;
		if(name.compare("SKY") == 0) id=2;
		if(name.compare("FOLIAGE") == 0) id=3;
		if(name.compare("CEMENT") == 0) id=4;
		if(name.compare("WINDOW") == 0) id=5;
		if(name.compare("PATH") == 0) id=6;
		if(name.compare("GRASS") == 0) id=7;
		if(id != 0) code.replace((id-1)*2,1,"1");
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




/* Prototype */
void load_file(string fileName, vector<Pattern> &p);
void createDB(vector<Pattern> &p);
void countNbP4Classes(vector <Pattern> &p, int *nb_inclass); 

int main() {

	string file_name("segmentation.all");
	
	//Creating the vectors of all patterns
	vector<Pattern> p(NOF_PATTERNS,Pattern());
	//Loading text files into vector array
	load_file(file_name,p);
	//Creating database with special ratio
	createDB(p);
		

	return 0;
}


void load_file(string fileName, vector<Pattern> &p) {
    
	string buf;
	int pos,num(0);
	int cnt(0);
	
	// Loading files
	cout << "loading " << fileName << endl;
	ifstream file;
	file.open(fileName.c_str(), ios::in);
	if(file.fail()) {
		cerr << "Cannot read : " << fileName << "." << endl;
		cerr << "Exiting now." << endl;
		file.close();
		exit(1);
	}


	//Parsing file
    	do {
        	getline(file, buf);
		//Check empty or comment line
        	if(buf[0] == '#' || buf.length() == 0) { //Do nothing
			pos = 0;
		}
		else {
			pos = buf.find_first_of(',');
			p[cnt].setCode(buf.substr(0,pos));
			p[cnt].setValues(buf.substr(pos+1,buf.length()));
            		cnt++;
		}        
    	} 
	while(!file.eof() );
	file.close();

	//Sorting all database in ascending id
	sort(p.begin(),p.end(), SortByCode()); 
}


void createDB(vector<Pattern> &p) {
	
	//Count how many pattern we have for each classes
	int count[] = {0,0,0,0,0,0,0}; 
	countNbP4Classes(p,count);
	for(int i(0); i<7; i++) cout << count[i] << " ";
	cout << endl;
	
	//Create the 3 output files
	ofstream fo_train("training", ios::trunc);
	ofstream fo_valid("valid",ios::trunc);
	ofstream fo_test("test",ios::trunc);

	//Maybe do it more modular (if all classes are not the same)
	int nb_train((int)(RATIO_TRAIN*(double)count[0]));
	int nb_valid((int)(RATIO_VALID*(double)count[0]));		
	int nb_test(count[0]-nb_train-nb_valid);
	cout << nb_train<<" "<< nb_valid <<" "<< nb_test << endl;

	//Fill first row in each datafiles
	fo_train << nb_train*NOF_CLASSES << " " << NOF_DIMS+NOF_CLASSES << endl;
	fo_valid << nb_valid*NOF_CLASSES << " " << NOF_DIMS+NOF_CLASSES << endl;
	fo_test  << nb_test*NOF_CLASSES  << " " << NOF_DIMS+NOF_CLASSES << endl;
	

	int j;
	for(int i(0); i< p.size(); i++) {
		j=i%count[0];
		//cout << j << endl;
		if(j<nb_train) fo_train << p[i] << endl; 
		else if(j< nb_train+nb_valid) fo_valid << p[i] << endl;
		else fo_test << p[i] << endl;
	}
	
	fo_train.close();	
	fo_valid.close();	
	fo_test.close();	

}  		

void countNbP4Classes(vector<Pattern> &p, int *nb_inclass) {
	for(int i(0); i<p.size(); i++) {
		nb_inclass[p[i].id-1]++;
	}
}
