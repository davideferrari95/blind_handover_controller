Eigen::Matrix<double, 4, 4> compute_UR5_direct_kinematic(Eigen::Matrix<double, 6, 1> q){

Eigen::Matrix<double, 4, 4> T;

double q0 = q(0,0);
double q1 = q(1,0);
double q2 = q(2,0);
double q3 = q(3,0);
double q4 = q(4,0);
double q5 = q(5,0);

T(0,0) = cos(q5)*(sin(q0)*sin(q4)+cos(q4)*(cos(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)-sin(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0))-sin(q5)*(sin(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)*1.0+cos(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0)*1.0;

T(0,1) = sin(q5)*(sin(q0)*sin(q4)+cos(q4)*(cos(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)-sin(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0))*-1.0-cos(q5)*(sin(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)*1.0+cos(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0)*1.0;

T(0,2) = cos(q4)*sin(q0)-sin(q4)*(cos(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)-sin(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0)*1.0;

T(0,3) = sin(q0)*1.0915E-1-cos(q0)*cos(q1)*4.25E-1+cos(q4)*sin(q0)*8.23E-2-sin(q4)*(cos(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)-sin(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*1.0)*8.23E-2+sin(q3)*(cos(q0)*cos(q1)*cos(q2)-cos(q0)*sin(q1)*sin(q2)*1.0)*9.465E-2+cos(q3)*(cos(q0)*cos(q1)*sin(q2)*1.0+cos(q0)*cos(q2)*sin(q1)*1.0)*9.465E-2-cos(q0)*cos(q1)*cos(q2)*3.9225E-1+cos(q0)*sin(q1)*sin(q2)*3.9225E-1;

T(1,0) = sin(q5)*(cos(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0-sin(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0)*-1.0-cos(q5)*(cos(q0)*sin(q4)*1.0+cos(q4)*(cos(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0+sin(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0)*1.0)*1.0;

T(1,1) = cos(q5)*(cos(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0-sin(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0)*-1.0+sin(q5)*(cos(q0)*sin(q4)*1.0+cos(q4)*(cos(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0+sin(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0)*1.0);

T(1,2) = cos(q0)*cos(q4)*-1.0+sin(q4)*(cos(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0+sin(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0);

T(1,3) = cos(q0)*-1.0915E-1-cos(q0)*cos(q4)*8.23E-2+cos(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*9.465E-2-cos(q1)*sin(q0)*4.25E-1-sin(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*9.465E-2+sin(q4)*(cos(q3)*(sin(q0)*sin(q1)*sin(q2)*1.0-cos(q1)*cos(q2)*sin(q0)*1.0)*1.0+sin(q3)*(cos(q1)*sin(q0)*sin(q2)*1.0+cos(q2)*sin(q0)*sin(q1)*1.0)*1.0)*8.23E-2+sin(q0)*sin(q1)*sin(q2)*3.9225E-1-cos(q1)*cos(q2)*sin(q0)*3.9225E-1;

T(2,0) = sin(q5)*(cos(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0)*1.0-sin(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)*1.0)+cos(q4)*cos(q5)*(cos(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)+sin(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0));

T(2,1) = cos(q5)*(cos(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0)*1.0-sin(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)*1.0)-cos(q4)*sin(q5)*(cos(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)+sin(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0))*1.0;

T(2,2) = sin(q4)*(cos(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)+sin(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0))*-1.0;

T(2,3) = sin(q1)*-4.25E-1-cos(q1)*sin(q2)*3.9225E-1-cos(q2)*sin(q1)*3.9225E-1-sin(q4)*(cos(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)+sin(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0))*8.23E-2-cos(q3)*(cos(q1)*cos(q2)*1.0-sin(q1)*sin(q2)*1.0)*9.465E-2+sin(q3)*(cos(q1)*sin(q2)*1.0+cos(q2)*sin(q1)*1.0)*9.465E-2+8.9159E-2;

T(3,0) = 0.0;

T(3,1) = 0.0;

T(3,2) = 0.0;

T(3,3) = 1.0;

return T;

}

