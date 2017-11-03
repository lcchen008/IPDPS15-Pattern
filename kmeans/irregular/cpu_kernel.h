#ifndef IRREGULAR_CPU_KERNEL_H_
#define IRREGULAR_CPU_KERNEL_H_

//This function is executed by CPU threads
void *compute_cpu(void *arg);
void *init_roc(void *arg);
void *mergetc(void *arg);

#endif
