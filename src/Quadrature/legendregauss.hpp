// Creates quadrature rule on an edge
//
// Author: Jerome Droniou (jerome.droniou@monash.edu)
//


#ifndef LEGENDREGAUSS_HPP_INCLUDED
#define LEGENDREGAUSS_HPP_INCLUDED
#include<stddef.h>

namespace Manicore {

/// \addtogroup Quadratures
///@{
  
  /// Compute the number of node, their location and the associated weight for a given degree of exactness
class LegendreGauss {
public:
    LegendreGauss(size_t doe /*!< Degree of exactness */); 
    ~LegendreGauss();

    size_t npts();
    /// Weight
    double wq(size_t i /*!< Node index */);
    /// Location
    double tq(size_t i /*!< Node index */);

private:
    void sub_rule_01();
    void sub_rule_02();
    void sub_rule_03();
    void sub_rule_04();
    void sub_rule_05();
    void sub_rule_06();
    void sub_rule_07();
    void sub_rule_08();
    void sub_rule_09();
    void sub_rule_10();
    void sub_rule_11();
	  void sub_rule_12();
		void sub_rule_13();
		void sub_rule_14();
		void sub_rule_15();
		void sub_rule_16();
		void sub_rule_17();
		void sub_rule_18();
		void sub_rule_19();
		void sub_rule_20();
		void sub_rule_21();

    size_t _doe;
    size_t _npts;
    double* _t;
    double* _w;
};
///@}
}

#endif 
