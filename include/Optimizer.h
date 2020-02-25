#ifndef  OPTIMIZER_H
#define  OPTIMIZER_H

#include "MapPoint.h"
#include "Frame.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"


namespace my_vo
{
  class Optimizer
  {
  public:
    int static PoseOptimization(Frame* pFrame);
    
    
  };
}




#endif