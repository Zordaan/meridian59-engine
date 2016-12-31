/*
-----------------------------------------------------------------------------
This source file is part of OGRE
    (Object-oriented Graphics Rendering Engine)
For the latest info, see http://www.ogre3d.org/

Copyright (c) 2000-2016 Torus Knot Software Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-----------------------------------------------------------------------------
*/
#ifndef __Vector4_H__
#define __Vector4_H__

#include "OgrePrerequisites.h"
#include "OgreVector3.h"

// Select alignment for Vector4 class
#if OGRE_SIMD_V4_32_SSE2
  #define OGRE_SIMD_V4_ALIGN ALIGN16
#elif OGRE_SIMD_V4_64_AVX
  #define OGRE_SIMD_V4_ALIGN ALIGN32
#else
  #define OGRE_SIMD_V4_ALIGN
#endif

namespace Ogre
{

    /** \addtogroup Core
    *  @{
    */
    /** \addtogroup Math
    *  @{
    */
    /** 4-dimensional homogeneous vector.
    */
    OGRE_SIMD_V4_ALIGN class _OgreExport Vector4
	{
	public:
        /** Anonymous union for access to vector data.
        */
        union
        {
            struct { Real x; Real y; Real z; Real w; };
            Real vals[4];

            #if OGRE_SIMD_V4_32_SSE2
            __m128  simd;
            #elif OGRE_SIMD_V4_64_AVX
            __m256d simd;
            #endif
        };

	public:
        /** Default constructor.
			@note
				It does <b>NOT</b> initialize the vector for efficiency.
        */
        FORCEINLINE Vector4()
        {
        }

		/** Constructor from four dedicated Real values. 
		*/
        FORCEINLINE Vector4(const Real fX, const Real fY, const Real fZ, const Real fW)
          #if OGRE_SIMD_V4_32_SSE2
            : simd(_mm_set_ps(fW, fZ, fY, fX)) { }
          #elif OGRE_SIMD_V4_32U_SSE2
            { _mm_storeu_ps(vals, _mm_set_ps(fW, fZ, fY, fX)); }
          #elif OGRE_SIMD_V4_64_AVX
            : simd(_mm256_set_pd(fW, fZ, fY, fX)) { }
          #elif OGRE_SIMD_V4_64U_AVX
            { _mm256_storeu_pd(vals, _mm256_set_pd(fW, fZ, fY, fX)); }
          #else
            : x(fX), y(fY), z(fZ), w(fW) { }
          #endif

        /** Constructor from Real array.
        */
        FORCEINLINE explicit Vector4(const Real afCoordinate[4])
          #if OGRE_SIMD_V4_32_SSE2
            : simd(_mm_loadu_ps(afCoordinate)) { }
          #elif OGRE_SIMD_V4_32U_SSE2
            { _mm_storeu_ps(vals, _mm_loadu_ps(afCoordinate)); }
          #elif OGRE_SIMD_V4_64_AVX
            : simd(_mm256_loadu_pd(afCoordinate)) { }
          #elif OGRE_SIMD_V4_64U_AVX
            { _mm256_storeu_pd(vals, _mm256_loadu_pd(afCoordinate)); }
          #else
            : x(afCoordinate[0]), y(afCoordinate[1]), z(afCoordinate[2]), w(afCoordinate[3]) { }
          #endif

		/** Constructor from int array.
		*/
        FORCEINLINE explicit Vector4(const int afCoordinate[4])
          #if OGRE_SIMD_V4_32_SSE2
          {
			// todo ?
            const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
            const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
            simd = _mm_cvtepi32_ps(b);                                  // convert ints to floats
          }
          #elif OGRE_SIMD_V4_32U_SSE2
          {
            const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
            const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
            _mm_storeu_ps(vals, _mm_cvtepi32_ps(b));                    // convert ints to floats
          }
          #elif OGRE_SIMD_V4_64_AVX
          {
            const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
            const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
            simd = _mm256_cvtepi32_pd(b);                               // convert ints to doubles
          }
          #elif OGRE_SIMD_V4_64U_AVX
          {
            const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
            const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
            _mm256_storeu_pd(vals, _mm256_cvtepi32_pd(b));              // convert ints to doubles
          }
          #else		
            : x((Real)afCoordinate[0]), 
              y((Real)afCoordinate[1]), 
              z((Real)afCoordinate[2]), 
              w((Real)afCoordinate[3])
          {
          }
          #endif

        /** Constructor from Real pointer.
        */
        FORCEINLINE explicit Vector4(Real* const r)
          #if OGRE_SIMD_V4_32_SSE2
            : simd(_mm_loadu_ps(r)) { }
          #elif OGRE_SIMD_V4_32U_SSE2
            { _mm_storeu_ps(vals, _mm_loadu_ps(r)); }
          #elif OGRE_SIMD_V4_64_AVX
            : simd(_mm256_loadu_pd(r)) { }
          #elif OGRE_SIMD_V4_64U_AVX
            { _mm256_storeu_pd(vals, _mm256_loadu_pd(r)); }
          #else
            : x(r[0]), y(r[1]), z(r[2]), w(r[3]) { }
          #endif

        /** Constructor from single scalar.
        */
        FORCEINLINE explicit Vector4(const Real scalar)
          #if OGRE_SIMD_V4_32_SSE2
            : simd(_mm_set1_ps(scalar)) { }
          #elif OGRE_SIMD_V4_32U_SSE2
            { _mm_storeu_ps(vals, _mm_set1_ps(scalar)); }
          #elif OGRE_SIMD_V4_64_AVX
            : simd(_mm256_set1_pd(scalar)) { }
          #elif OGRE_SIMD_V4_64U_AVX
            { _mm256_storeu_pd(vals, _mm256_set1_pd(scalar)); }
          #else
			: x(scalar), y(scalar), z(scalar), w(scalar) { }
          #endif

        FORCEINLINE explicit Vector4(const Vector3& rhs)
            : x(rhs.x), y(rhs.y), z(rhs.z), w(1.0f)
		{
		}

        #if OGRE_SIMD_V4_32_SSE2
          // Aligned SSE Constructor
          FORCEINLINE Vector4(const __m128 values) : simd(values) { }
        #elif OGRE_SIMD_V4_32U_SSE2
          // Unaligned SSE Constructor
          FORCEINLINE Vector4(const __m128 values) { _mm_storeu_ps(vals, values); }
        #elif OGRE_SIMD_V4_64_AVX
          // Aligned AVX Constructor
          FORCEINLINE Vector4(const __m256d values) : simd(values) { }
        #elif OGRE_SIMD_V4_64U_AVX
          // Unaligned AVX Constructor
          FORCEINLINE Vector4(const __m256d values) { _mm256_storeu_pd(vals, values); }
        #endif

		/** Swizzle-like narrowing operations
		*/
        FORCEINLINE Vector3 xyz() const
		{
			return Vector3(x, y, z);
		}
        FORCEINLINE Vector2 xy() const
		{
			return Vector2(x, y);
		}

        /** Exchange the contents of this vector with another. 
		*/
        FORCEINLINE void swap(Vector4& other)
		{
          #if OGRE_SIMD_V4_32_SSE2
            const __m128 a = simd;         // load a from this
            const __m128 b = other.simd;   // load b from other
            simd = b;                      // save b to this
            other.simd = a;                // save a to other
          #elif OGRE_SIMD_V4_32U_SSE2
            const __m128 a = _mm_loadu_ps(vals);       // load a from this
            const __m128 b = _mm_loadu_ps(other.vals); // load b from other
            _mm_storeu_ps(vals, b);                    // save b to this
            _mm_storeu_ps(other.vals, a);              // save a to other	
          #elif OGRE_SIMD_V4_64_AVX
            const __m256d a = simd;        // load a from this
            const __m256d b = other.simd;  // load b from other
            simd = b;                      // save b to this
            other.simd = a;                // save a to other
          #elif OGRE_SIMD_V4_64U_AVX
            const __m256d a = _mm256_loadu_pd(vals);       // load a from this
            const __m256d b = _mm256_loadu_pd(other.vals); // load b from other
            _mm256_storeu_pd(vals, b);                     // save b to this
            _mm256_storeu_pd(other.vals, a);               // save a to other
          #else
			std::swap(x, other.x);
			std::swap(y, other.y);
			std::swap(z, other.z);
			std::swap(w, other.w);
          #endif
        }
    
        FORCEINLINE Real operator [] ( const size_t i ) const
        {
            assert( i < 4 );

            return *(&x+i);
        }

        FORCEINLINE Real& operator [] ( const size_t i )
        {
            assert( i < 4 );

            return *(&x+i);
        }

        /// Pointer accessor for direct copying
        FORCEINLINE Real* ptr()
        {
            return &x;
        }
        /// Pointer accessor for direct copying
        FORCEINLINE const Real* ptr() const
        {
            return &x;
        }

        /** Assigns the value of the other vector.
            @param
                rkVector The other vector
        */
        FORCEINLINE Vector4& operator = ( const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2 || OGRE_SIMD_V4_64_AVX
            simd = rkVector.simd;
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_loadu_ps(rkVector.vals));
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_loadu_pd(rkVector.vals));
          #else
            x = rkVector.x;
            y = rkVector.y;
            z = rkVector.z;
            w = rkVector.w;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator = ( const Real fScalar)
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_set1_ps(fScalar);
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_set1_ps(fScalar));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_set1_pd(fScalar);
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_set1_pd(fScalar));
          #else
            x = fScalar;
            y = fScalar;
            z = fScalar;
            w = fScalar;
          #endif
            return *this;
        }

        FORCEINLINE bool operator == ( const Vector4& rkVector ) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            const __m128  a = _mm_load_ps(ptr());          // load this
            const __m128  b = _mm_load_ps(rkVector.ptr()); // load other
            const __m128  c = _mm_cmpeq_ps(a, b);          // compare all components
            const __m128i d = _mm_castps_si128(c);         // cast from float to int (no-op)
            const int     e = _mm_movemask_epi8(d);        // combine some bits from all registers
            const bool    f = e == 0xFFFF;                 // check all equal
            return f;
          #elif OGRE_SIMD_V4_64_AVX
            const __m256d a = simd;                            // load this
            const __m256d b = rkVector.simd;                   // load other
            const __m256d c = _mm256_cmp_pd(a, b, _CMP_EQ_OQ); // compare all components			
            const __m256i d = _mm256_castpd_si256(c);          // cast from double to int64 (no-op)
            const int     e = _mm256_movemask_epi8(d);         // combine some bits from all registers
            const bool    f = e == 0xFFFFFFFF;                 // check all equal
            return f;
          #else
            return ( x == rkVector.x &&
                y == rkVector.y &&
                z == rkVector.z &&
                w == rkVector.w );
          #endif
        }

        FORCEINLINE bool operator != ( const Vector4& rkVector ) const
        {
          #if OGRE_SIMD_V4_32_SSE2
			const __m128  a = simd;                        // load this
			const __m128  b = rkVector.simd;               // load other
			const __m128  c = _mm_cmpeq_ps(a, b);          // compare all components
            const __m128i d = _mm_castps_si128(c);         // cast from float to int (no-op)
            const int     e = _mm_movemask_epi8(d);        // combine some bits from all registers
            const bool    f = e != 0xFFFF;                 // check not all equal
            return f;
          #elif OGRE_SIMD_V4_64_AVX
            const __m256d a = simd;                            // load this
            const __m256d b = rkVector.simd;                   // load other
            const __m256d c = _mm256_cmp_pd(a, b, _CMP_EQ_OQ); // compare all components			
            const __m256i d = _mm256_castpd_si256(c);          // cast from double to int64 (no-op)
            const int     e = _mm256_movemask_epi8(d);         // combine some bits from all registers
            const bool    f = e != 0xFFFFFFFF;                 // check not all equal
            return f;
          #else
            return ( x != rkVector.x ||
                y != rkVector.y ||
                z != rkVector.z ||
                w != rkVector.w );
          #endif
        }

        FORCEINLINE Vector4& operator = (const Vector3& rhs)
        {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            w = 1.0f;
            return *this;
        }

        // arithmetic operations
        FORCEINLINE Vector4 operator + ( const Vector4& rkVector ) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_add_ps(simd, rkVector.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_add_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_add_pd(simd, rkVector.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_add_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            return Vector4(
                x + rkVector.x,
                y + rkVector.y,
                z + rkVector.z,
                w + rkVector.w);
          #endif
        }

        FORCEINLINE Vector4 operator - ( const Vector4& rkVector ) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_sub_ps(simd, rkVector.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_sub_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_sub_pd(simd, rkVector.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_sub_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            return Vector4(
                x - rkVector.x,
                y - rkVector.y,
                z - rkVector.z,
                w - rkVector.w);
          #endif
        }

        FORCEINLINE Vector4 operator * ( const Real fScalar ) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_mul_ps(simd, _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_mul_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_mul_pd(simd, _mm256_set1_pd(fScalar)));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_mul_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
		  #else
            return Vector4(
                x * fScalar,
                y * fScalar,
                z * fScalar,
                w * fScalar);
		  #endif
        }
		
        FORCEINLINE Vector4 operator * ( const Vector4& rhs) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_mul_ps(simd, rhs.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_mul_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rhs.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_mul_pd(simd, rhs.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_mul_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rhs.vals)));
		  #else
            return Vector4(
                rhs.x * x,
                rhs.y * y,
                rhs.z * z,
                rhs.w * w);
		  #endif
        }

        FORCEINLINE Vector4 operator / ( const Real fScalar ) const
        {
          assert(fScalar != 0.0);

          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_div_ps(simd, _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_div_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_div_pd(simd, _mm256_set1_pd(fScalar)));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_div_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
          #else
            Real fInv = 1.0f / fScalar;

            return Vector4(
                x * fInv,
                y * fInv,
                z * fInv,
                w * fInv);
          #endif
        }

        FORCEINLINE Vector4 operator / ( const Vector4& rhs) const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_div_ps(simd, rhs.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_div_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rhs.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_div_pd(simd, rhs.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_div_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rhs.vals)));
          #else
            return Vector4(
                x / rhs.x,
                y / rhs.y,
                z / rhs.z,
                w / rhs.w);
          #endif
        }

        FORCEINLINE const Vector4& operator + () const
        {
            return *this;
        }

        FORCEINLINE Vector4 operator - () const
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_xor_ps(simd, SIGNMASK_SSE));
          #elif OGRE_SIMD_V4_32U_SSE2
            const __m128 a = _mm_loadu_ps(vals);          // load values
            const __m128 b = _mm_xor_ps(a, SIGNMASK_SSE); // flip sign bit
            return Vector4(b);
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_xor_pd(simd, SIGNMASK_AVX));
          #elif OGRE_SIMD_V4_64U_AVX
            const __m256d a = _mm256_loadu_pd(vals);          // load values
            const __m256d b = _mm256_xor_pd(a, SIGNMASK_AVX); // flip sign bit
            return Vector4(b);
          #else
            return Vector4(-x, -y, -z, -w);
          #endif
        }

        FORCEINLINE friend Vector4 operator * ( const Real fScalar, const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_mul_ps(_mm_set1_ps(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_mul_ps(_mm_set1_ps(fScalar), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_mul_pd(_mm256_set1_pd(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_mul_pd(_mm256_set1_pd(fScalar), _mm256_loadu_pd(rkVector.vals)));
          #else
            return Vector4(
                fScalar * rkVector.x,
                fScalar * rkVector.y,
                fScalar * rkVector.z,
                fScalar * rkVector.w);
          #endif
        }

        FORCEINLINE friend Vector4 operator / ( const Real fScalar, const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_div_ps(_mm_set1_ps(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_div_ps(_mm_set1_ps(fScalar), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_div_pd(_mm256_set1_pd(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_div_pd(_mm256_set1_pd(fScalar), _mm256_loadu_pd(rkVector.vals)));
          #else
            return Vector4(
                fScalar / rkVector.x,
                fScalar / rkVector.y,
                fScalar / rkVector.z,
                fScalar / rkVector.w);
          #endif
        }

        FORCEINLINE friend Vector4 operator + (const Vector4& lhs, const Real rhs)
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_add_ps(lhs.simd, _mm_set1_ps(rhs)));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_add_ps(_mm_loadu_ps(lhs.vals), _mm_set1_ps(rhs)));
          #elif OGRE_SIMD_V4_64_AVX
			return Vector4(_mm256_add_pd(lhs.simd, _mm256_set1_pd(rhs)));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_add_pd(_mm256_loadu_pd(lhs.vals), _mm256_set1_pd(rhs)));
          #else
            return Vector4(
                lhs.x + rhs,
                lhs.y + rhs,
                lhs.z + rhs,
                lhs.w + rhs);
          #endif
        }

        FORCEINLINE friend Vector4 operator + (const Real lhs, const Vector4& rhs)
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_add_ps(_mm_set1_ps(lhs), rhs.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_add_ps(_mm_set1_ps(lhs), _mm_loadu_ps(rhs.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_add_pd(_mm256_set1_pd(lhs), rhs.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_add_pd(_mm256_set1_pd(lhs), _mm256_loadu_pd(rhs.vals)));
          #else
            return Vector4(
                lhs + rhs.x,
                lhs + rhs.y,
                lhs + rhs.z,
                lhs + rhs.w);
          #endif
        }

        FORCEINLINE friend Vector4 operator - (const Vector4& lhs, Real rhs)
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_sub_ps(lhs.simd, _mm_set1_ps(rhs)));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_sub_ps(_mm_loadu_ps(lhs.vals), _mm_set1_ps(rhs)));
          #elif OGRE_SIMD_V4_64_AVX
            return Vector4(_mm256_sub_pd(lhs.simd, _mm256_set1_pd(rhs)));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_sub_pd(_mm256_loadu_pd(lhs.vals), _mm256_set1_pd(rhs)));
          #else
            return Vector4(
                lhs.x - rhs,
                lhs.y - rhs,
                lhs.z - rhs,
                lhs.w - rhs);
          #endif
        }

        FORCEINLINE friend Vector4 operator - (const Real lhs, const Vector4& rhs)
        {
          #if OGRE_SIMD_V4_32_SSE2
            return Vector4(_mm_sub_ps(_mm_set1_ps(lhs), rhs.simd));
          #elif OGRE_SIMD_V4_32U_SSE2
            return Vector4(_mm_sub_ps(_mm_set1_ps(lhs), _mm_loadu_ps(rhs.vals)));
          #elif OGRE_SIMD_V4_64_AVX
			return Vector4(_mm256_sub_pd(_mm256_set1_pd(lhs), rhs.simd));
          #elif OGRE_SIMD_V4_64U_AVX
            return Vector4(_mm256_sub_pd(_mm256_set1_pd(lhs), _mm256_loadu_pd(rhs.vals)));
          #else
            return Vector4(
                lhs - rhs.x,
                lhs - rhs.y,
                lhs - rhs.z,
                lhs - rhs.w);
          #endif
        }

        // arithmetic updates
        FORCEINLINE Vector4& operator += ( const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_add_ps(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_add_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_add_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_add_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            x += rkVector.x;
            y += rkVector.y;
            z += rkVector.z;
            w += rkVector.w;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator -= ( const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_sub_ps(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_sub_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_sub_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_sub_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            x -= rkVector.x;
            y -= rkVector.y;
            z -= rkVector.z;
            w -= rkVector.w;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator *= ( const Real fScalar )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_mul_ps(simd, _mm_set1_ps(fScalar));
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_mul_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_mul_pd(simd, _mm256_set1_pd(fScalar));
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_mul_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
          #else
            x *= fScalar;
            y *= fScalar;
            z *= fScalar;
            w *= fScalar;
          #endif
			return *this;
        }

        FORCEINLINE Vector4& operator += ( const Real fScalar )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_add_ps(simd, _mm_set1_ps(fScalar));
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_add_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_add_pd(simd, _mm256_set1_pd(fScalar));
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_add_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
          #else
            x += fScalar;
            y += fScalar;
            z += fScalar;
            w += fScalar;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator -= ( const Real fScalar )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_sub_ps(simd, _mm_set1_ps(fScalar));
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_sub_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_sub_pd(simd, _mm256_set1_pd(fScalar));
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_sub_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
          #else
            x -= fScalar;
            y -= fScalar;
            z -= fScalar;
            w -= fScalar;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator *= ( const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_mul_ps(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_mul_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_mul_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_64U_AVX
			_mm256_storeu_pd(vals, _mm256_mul_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            x *= rkVector.x;
            y *= rkVector.y;
            z *= rkVector.z;
            w *= rkVector.w;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator /= ( const Real fScalar )
        {
          assert( fScalar != 0.0 );

          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_div_ps(simd, _mm_set1_ps(fScalar));
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_div_ps(_mm_loadu_ps(vals), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_div_pd(simd, _mm256_set1_pd(fScalar));
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_div_pd(_mm256_loadu_pd(vals), _mm256_set1_pd(fScalar)));
          #else
            const Real fInv = 1.0f / fScalar; // TODO ?
            x *= fInv;
            y *= fInv;
            z *= fInv;
            w *= fInv;
          #endif
            return *this;
        }

        FORCEINLINE Vector4& operator /= ( const Vector4& rkVector )
        {
          #if OGRE_SIMD_V4_32_SSE2
            simd = _mm_div_ps(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_32U_SSE2
            _mm_storeu_ps(vals, _mm_div_ps(_mm_loadu_ps(vals), _mm_loadu_ps(rkVector.vals)));
          #elif OGRE_SIMD_V4_64_AVX
            simd = _mm256_div_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V4_64U_AVX
            _mm256_storeu_pd(vals, _mm256_div_pd(_mm256_loadu_pd(vals), _mm256_loadu_pd(rkVector.vals)));
          #else
            x /= rkVector.x;
            y /= rkVector.y;
            z /= rkVector.z;
            w /= rkVector.w;
          #endif
            return *this;
        }

        /** Calculates the dot (scalar) product of this vector with another.
            @param
                vec Vector with which to calculate the dot product (together
                with this one).
            @return
                A float representing the dot product value.
        */
        FORCEINLINE Real dotProduct(const Vector4& vec) const
        {
          #if OGRE_SIMD_V4_32_SSE41
            return _mm_dp_ps(simd, vec.simd, 0xFF).m128_f32[0];
          #elif OGRE_SIMD_V4_32U_SSE41
            return _mm_dp_ps(_mm_loadu_ps(vals), _mm_loadu_ps(vec.vals), 0xFF).m128_f32[0];
          #else
            return x * vec.x + y * vec.y + z * vec.z + w * vec.w;
          #endif
        }

        /// Check whether this vector contains valid values
        FORCEINLINE bool isNaN() const
        {
          #if OGRE_SIMD_V4_32_SSE2
            const __m128  a = simd;
            const __m128  b = _mm_cmpeq_ps(a, a);    // NaN has: (NaN == NaN) = false
            const __m128i c = _mm_castps_si128(b);   // cast from float to int (no-op)
            const int     d = _mm_movemask_epi8(c);  // combine some bits from all registers
            const bool    e = d != 0xFFFF;           // check not all equal
            return e;
          #elif OGRE_SIMD_V4_32U_SSE2
            const __m128  a = _mm_loadu_ps(vals);
            const __m128  b = _mm_cmpeq_ps(a, a);    // NaN has: (NaN == NaN) = false
            const __m128i c = _mm_castps_si128(b);   // cast from float to int (no-op)
            const int     d = _mm_movemask_epi8(c);  // combine some bits from all registers
            const bool    e = d != 0xFFFF;           // check not all equal
            return e;
          #else
            return Math::isNaN(x) || Math::isNaN(y) || Math::isNaN(z) || Math::isNaN(w);
          #endif
        }

        /** Function for writing to a stream.
        */
        FORCEINLINE _OgreExport friend std::ostream& operator <<
            ( std::ostream& o, const Vector4& v )
        {
            o << "Vector4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
            return o;
        }
        // special
        static const Vector4 ZERO;

      #if OGRE_SIMD_V4_32_SSE2 || OGRE_SIMD_V4_32U_SSE2
		static const __m128 SIGNMASK_SSE;
      #elif OGRE_SIMD_V4_64_AVX || OGRE_SIMD_V4_64U_AVX
		static const __m256d SIGNMASK_AVX;
      #endif

    };
    /** @} */
    /** @} */

}
#endif

