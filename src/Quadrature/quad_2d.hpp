// Creates quadrature rule in a cell
//
// Author: Jerome Droniou (jerome.droniou@monash.edu)
//

/*
*
*  This library was developed around HHO methods, although some parts of it have a more
* general purpose. If you use this code or part of it in a scientific publication, 
* please mention the following book as a reference for the underlying principles
* of HHO schemes:
*
* The Hybrid High-Order Method for Polytopal Meshes: Design, Analysis, and Applications. 
*  D. A. Di Pietro and J. Droniou. Modeling, Simulation and Applications, vol. 19. 
*  Springer International Publishing, 2020, xxxi + 525p. doi: 10.1007/978-3-030-37203-3. 
*  url: https://hal.archives-ouvertes.fr/hal-02151813.
*
*/


#ifndef QUAD_2D_HPP
#define QUAD_2D_HPP

#include <cstddef>

namespace Manicore {

/// \addtogroup Quadratures
///@{

/**
* @brief Wrapper for dunavant quadrature rules
*/
class QuadRuleTriangle {
    static constexpr size_t max_doe = 20;

public:
    /**
    * @brief Default constructor
    *
    * @param doe degrees of exactness (e.g. how many points for approximating
    *integral
    */
    QuadRuleTriangle(size_t doe);
    ~QuadRuleTriangle();

    size_t nq();             /// <\brief returns number of points
    double xq(size_t i);  /// <\brief
    double yq(size_t i);  /// <\brief
    double wq(size_t i);  /// <\brief get the weight for a given point
    void setup(double xV[], double yV[]);  /// <\brief setup the quad rule given
                                           /// vertex coords

private:
    size_t _npts;
    double* _xy;
    double* _w;
    double* _xyphys;
    double area;
};

///@}
}
#endif /* QUAD2D_HPP */
