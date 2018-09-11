/*
 * from https://github.com/jMonkeyEngine/jmonkeyengine/blob/master/jme3-core/src/main/resources/Common/OpenCL/Matrix4f.clh
*/


typedef float16 mat4;

//Returns the zero matrix
inline mat4 mat4Zero() {
	return (float16)(0);
}

//Returns the identity matrix
inline mat4 mat4Identity() {
	return (float16)
		(1, 0, 0, 0,
		 0, 1, 0, 0,
		 0, 0, 1, 0,
		 0, 0, 0, 1);
}

//Sets the i-th row (0-based)
inline mat4 mat4SetRow(mat4 mat, int i, float4 row) {
	if (i==0) mat.s0123 = row;
	else if (i==1) mat.s4567 = row;
	else if (i==2) mat.s89ab = row;
	else mat.scdef = row;
	return mat;
}

inline mat4 mat4FromRows(float4 row1, float4 row2, float4 row3, float4 row4) {
	return (float16) (row1, row2, row3, row4);
}

inline mat4 mat4FromColumns(float4 col1, float4 col2, float4 col3, float4 col4) {
	return (float16)
		(col1.x, col2.x, col3.x, col4.x,
		 col1.y, col2.y, col3.y, col4.y,
		 col1.z, col2.z, col3.z, col4.z,
		 col1.w, col2.w, col3.w, col4.w);
}

inline mat4 mat4FromDiagonal(float4 diag) {
	return (float16)
		(diag.x, 0, 0, 0,
		 0, diag.y, 0, 0,
		 0, 0, diag.z, 0,
		 0, 0, 0, diag.w);
}

inline mat4 mat4Transpose(mat4 mat) {
	return mat.s048c159d26ae37bf; //magic
}

//Multiplies the two matrices A and B
inline mat4 mat4Mult(mat4 A, mat4 B) {
	return (float16) (
		dot(A.s0123, B.s048c),
		dot(A.s0123, B.s159d),
		dot(A.s0123, B.s26ae),
		dot(A.s0123, B.s37bf),

		dot(A.s4567, B.s048c),
		dot(A.s4567, B.s159d),
		dot(A.s4567, B.s26ae),
		dot(A.s4567, B.s37bf),

		dot(A.s89ab, B.s048c),
		dot(A.s89ab, B.s159d),
		dot(A.s89ab, B.s26ae),
		dot(A.s89ab, B.s37bf),

		dot(A.scdef, B.s048c),
		dot(A.scdef, B.s159d),
		dot(A.scdef, B.s26ae),
		dot(A.scdef, B.s37bf)
	);
}

mat4 mat4Invert(mat4 mat) {
	float fA0 = mat.s0 * mat.s5 - mat.s1 * mat.s4;
	float fA1 = mat.s0 * mat.s6 - mat.s2 * mat.s4;
	float fA2 = mat.s0 * mat.s7 - mat.s3 * mat.s4;
	float fA3 = mat.s1 * mat.s6 - mat.s2 * mat.s5;
	float fA4 = mat.s1 * mat.s7 - mat.s3 * mat.s5;
	float fA5 = mat.s2 * mat.s7 - mat.s3 * mat.s6;
	float fB0 = mat.s8 * mat.sd - mat.s9 * mat.sc;
	float fB1 = mat.s8 * mat.se - mat.sa * mat.sc;
	float fB2 = mat.s8 * mat.sf - mat.sb * mat.sc;
	float fB3 = mat.s9 * mat.se - mat.sa * mat.sd;
	float fB4 = mat.s9 * mat.sf - mat.sb * mat.sd;
	float fB5 = mat.sa * mat.sf - mat.sb * mat.se;
	float fDet = fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;

	if (fabs(fDet) <= 0.000001f) {
		return mat4Zero();
	}

	mat4 store;
	store.s0 = +mat.s5 * fB5 - mat.s6 * fB4 + mat.s7 * fB3;
	store.s4 = -mat.s4 * fB5 + mat.s6 * fB2 - mat.s7 * fB1;
	store.s8 = +mat.s4 * fB4 - mat.s5 * fB2 + mat.s7 * fB0;
	store.sc = -mat.s4 * fB3 + mat.s5 * fB1 - mat.s6 * fB0;
	store.s1 = -mat.s1 * fB5 + mat.s2 * fB4 - mat.s3 * fB3;
	store.s5 = +mat.s0 * fB5 - mat.s2 * fB2 + mat.s3 * fB1;
	store.s9 = -mat.s0 * fB4 + mat.s1 * fB2 - mat.s3 * fB0;
	store.sd = +mat.s0 * fB3 - mat.s1 * fB1 + mat.s2 * fB0;
	store.s2 = +mat.sd * fA5 - mat.se * fA4 + mat.sf * fA3;
	store.s6 = -mat.sc * fA5 + mat.se * fA2 - mat.sf * fA1;
	store.sa = +mat.sc * fA4 - mat.sd * fA2 + mat.sf * fA0;
	store.se = -mat.sc * fA3 + mat.sd * fA1 - mat.se * fA0;
	store.s3 = -mat.s9 * fA5 + mat.sa * fA4 - mat.sb * fA3;
	store.s7 = +mat.s8 * fA5 - mat.sa * fA2 + mat.sb * fA1;
	store.sb = -mat.s8 * fA4 + mat.s9 * fA2 - mat.sb * fA0;
	store.sf = +mat.s8 * fA3 - mat.s9 * fA1 + mat.sa * fA0;

	store /= fDet;

	return store;
}
