#ifndef CCSE_H
#define CCSE_H

#include "../proNet.h"

/*****
 * CCSE
 * **************************************************************/

class CCSE {

    public:
        
        CCSE();
        ~CCSE();
        
        proNet pnet;

        // parameters
        int dim;                // representation dimensions
        vector< vector<double> > w_vertexU;
        vector< vector<double> > w_vertexI;
        vector< vector<double> > w_contextU;
        vector< vector<double> > w_contextI;
        vector< vector<double> > w_contextUU;
        vector< vector<double> > w_contextII;
        vector< vector<double> > w_vertexP;
        vector< vector<double> > w_contextP;

        // data function
        void LoadEdgeList(string, bool);
        void LoadFieldMeta(string);
        void SaveWeights(string);
        
        // model function
        void Init(int);
        void Train(int, int, double, int, double, double);

};


#endif
