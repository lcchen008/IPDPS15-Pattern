#ifndef IRREGULAR_REDUCTION_ARRAY_H_
#define IRREGULAR_REDUCTION_ARRAY_H_
#include "../lib/common.h"

class reduction_array
{
protected:
	int unit_size_;
	IRIndex num_elms_;
	int *global_id_;    //indicates the position of each element in array_
	void *array_;
public:
	IRIndex num_elms(){return num_elms_;}
	int *global_id(){return global_id_}
	void *array(){return array_;}
};

//partial reduction result from peers, to be reduced to local array
class incoming_reduction_array:public reduction_array
{
protected:
	int rank_; //the rank of the sender	
public:
	int rank(){return rank_;}
};

//partial reduction result to peers, to be sent to remote
class outgoing_reduction_array:public reduction_array
{
protected:
	int rank_; //the rank of the sender	
public:
	int rank(){return rank_;}
};


#endif
