#define IMUL32(a,b) ((a)*(b))
#define UMUL32(a,b) ((a)*(b))
#define UMUL16_16(a,b) ((a)*(b))

#define celt_word16 float
#define celt_word32 float

#define celt_sig float
#define celt_norm float
#define celt_ener float
#define celt_pgain float
#define celt_mask float

#define UADD32(a,b) ((a)+(b))
#define USUB32(a,b) ((a)-(b))


#define Q15ONE 1.0f
#define Q30ONE 1.0f

#define NORM_SCALING 1.f
#define NORM_SCALING_1 1.f
#define ENER_SCALING 1.f
#define ENER_SCALING_1 1.f
#define PGAIN_SCALING 1.f
#define PGAIN_SCALING_1 1.f

#define DB_SCALING 1.f
#define DB_SCALING_1 1.f

#define EPSILON 1e-15f
#define VERY_SMALL 1e-15f
#define VERY_LARGE32 1e15f
#define VERY_LARGE16 1e15f
#define Q15_ONE 1.f
#define Q15_ONE_1 1.f

#define QCONST16(x,bits) (x)
#define QCONST32(x,bits) (x)

#define NEG16(x) (-(x))
#define NEG32(x) (-(x))
#define EXTRACT16(x) (x)
#define EXTEND32(x) (x)
#define SHR16(a,shift) (a)
#define SHL16(a,shift) (a)
#define SHR32(a,shift) (a)
#define SHL32(a,shift) (a)
#define PSHR16(a,shift) (a)
#define PSHR32(a,shift) (a)
#define VSHR32(a,shift) (a)
#define SATURATE16(x,a) (x)
#define SATURATE32(x,a) (x)

#define PSHR(a,shift)   (a)
#define SHR(a,shift)    (a)
#define SHL(a,shift)    (a)
#define SATURATE(x,a)   (x)

#define ROUND16(a,shift)  (a)
#define HALF32(x)       (.5f*(x))

#define ADD16(a,b) ((a)+(b))
#define SUB16(a,b) ((a)-(b))
#define ADD32(a,b) ((a)+(b))
#define SUB32(a,b) ((a)-(b))
#define MULT16_16_16(a,b)     ((a)*(b))
#define MULT16_16(a,b)     ((a)*(b))
#define MAC16_16(c,a,b)     ((c)+(a)*(b))

#define MULT16_32_Q11(a,b)     ((a)*(b))
#define MULT16_32_Q13(a,b)     ((a)*(b))
#define MULT16_32_Q14(a,b)     ((a)*(b))
#define MULT16_32_Q15(a,b)     ((a)*(b))
#define MULT16_32_Q16(a,b)     ((a)*(b))
#define MULT16_32_P15(a,b)     ((a)*(b))

#define MULT32_32_Q31(a,b)     ((a)*(b))

#define MAC16_32_Q11(c,a,b)     ((c)+(a)*(b))
#define MAC16_32_Q15(c,a,b)     ((c)+(a)*(b))

#define MAC16_16_Q11(c,a,b)     ((c)+(a)*(b))
#define MAC16_16_Q13(c,a,b)     ((c)+(a)*(b))
#define MAC16_16_P13(c,a,b)     ((c)+(a)*(b))
#define MULT16_16_Q11_32(a,b)     ((a)*(b))
#define MULT16_16_Q13(a,b)     ((a)*(b))
#define MULT16_16_Q14(a,b)     ((a)*(b))
#define MULT16_16_Q15(a,b)     ((a)*(b))
#define MULT16_16_P15(a,b)     ((a)*(b))
#define MULT16_16_P13(a,b)     ((a)*(b))
#define MULT16_16_P14(a,b)     ((a)*(b))

#define DIV32_16(a,b)     ((a)/(b))
#define PDIV32_16(a,b)     ((a)/(b))
#define DIV32(a,b)     ((a)/(b))
#define PDIV32(a,b)     ((a)/(b))

#define PRINT_MIPS(x)
