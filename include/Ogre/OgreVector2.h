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
#ifndef __Vector2_H__
#define __Vector2_H__


#include "OgrePrerequisites.h"
#include "OgreMath.h"

// Select alignment for Vector2 class
#if OGRE_SIMD_V2_32_SSE2
  #define OGRE_SIMD_V2_ALIGN ALIGN8
#elif OGRE_SIMD_V2_64_SSE2
  #define OGRE_SIMD_V2_ALIGN ALIGN16
#else
  #define OGRE_SIMD_V2_ALIGN
#endif

namespace Ogre
{

    /** \addtogroup Core
    *  @{
    */
    /** \addtogroup Math
    *  @{
    */
    /** Standard 2-dimensional vector.
        @remarks
            A direction in 2D space represented as distances along the 2
            orthogonal axes (x, y). Note that positions, directions and
            scaling factors can be represented by a vector, depending on how
            you interpret the values.
    */
    OGRE_SIMD_V2_ALIGN class _OgreExport Vector2
    {
    public:
        /** Anonymous union for access to vector data.
        */
        union
        {
            struct { Real x; Real y; };
            Real vals[2];

            #if OGRE_SIMD_V2_64_SSE2
            __m128d simd;
            #endif
        };

    public:
        /** Default constructor.
            @note
                It does <b>NOT</b> initialize the vector for efficiency.
        */
        inline Vector2()
        {
        }

        inline Vector2(const Real fX, const Real fY )
          #if OGRE_SIMD_V2_64_SSE2
            : simd(_mm_set_pd(fY, fX)) { }
          #elif OGRE_SIMD_V2_64U_SSE2
            { _mm_storeu_pd(vals, _mm_set_pd(fY, fX)); }
          #else
            : x( fX ), y( fY ) { }
          #endif

        inline explicit Vector2( const Real scalar )
          #if OGRE_SIMD_V2_64_SSE2
            : simd(_mm_set1_pd(scalar)) { }
          #elif OGRE_SIMD_V2_64U_SSE2
            { _mm_storeu_pd(vals, _mm_set1_pd(scalar)); }
          #else
            : x( scalar), y( scalar ) { }
          #endif

        inline explicit Vector2(const Real afCoordinate[2])
          #if OGRE_SIMD_V2_32_SSE2
            { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)afCoordinate)); }
          #elif OGRE_SIMD_V2_64_SSE2
            : simd(_mm_loadu_pd(afCoordinate)) { }
          #elif OGRE_SIMD_V2_64U_SSE2
            { _mm_storeu_pd(vals, _mm_loadu_pd(afCoordinate)); }
          #else
            : x( afCoordinate[0] ), y( afCoordinate[1] ) { }
          #endif

        inline explicit Vector2( const int afCoordinate[2] )
          #if OGRE_SIMD_V2_32_SSE2
            { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_cvtepi32_ps(_mm_loadl_epi64((__m128i*)afCoordinate)))); }
          #elif OGRE_SIMD_V2_64_SSE2
            : simd(_mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)afCoordinate))) { }
          #elif OGRE_SIMD_V2_64U_SSE2
            { _mm_storeu_pd(vals, _mm_cvtepi32_pd(_mm_loadl_epi64((__m128i*)afCoordinate))); }
          #else
            : x((Real)afCoordinate[0]), y((Real)afCoordinate[1]) { }
          #endif

        inline explicit Vector2( Real* const r )
          #if OGRE_SIMD_V2_32_SSE2
            { _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)r)); }
          #elif OGRE_SIMD_V2_64_SSE2
            : simd(_mm_loadu_pd(r)) { }
          #elif OGRE_SIMD_V2_64U_SSE2
            { _mm_storeu_pd(vals, _mm_loadu_pd(r)); }
          #else
            : x( r[0] ), y( r[1] ) { }
          #endif

        #if OGRE_SIMD_V2_32_SSE2
          // SSE Constructor
          FORCEINLINE Vector2(const __m128 values) { _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(values)); }
        #elif OGRE_SIMD_V2_64_SSE2
          // Aligned SSE Constructor
          FORCEINLINE Vector2(const __m128d values) : simd(values) { }
		#elif OGRE_SIMD_V2_64U_SSE2
          // Unaligned SSE Constructor
          FORCEINLINE Vector2(const __m128d values) { _mm_storeu_pd(vals, values); }
        #endif

        /** Exchange the contents of this vector with another. 
        */
        inline void swap(Vector2& other)
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128i a = _mm_loadl_epi64((__m128i*)vals);                       // load this to low 64bits
            const __m128  b = _mm_loadh_pi(_mm_castsi128_ps(a), (__m64*)other.vals); // load other to high 64bits		
            _mm_storel_epi64((__m128i*)other.vals, _mm_castps_si128(b));             // save low 64bits to other
            _mm_storeh_pi((__m64*)vals, b);                                          // save high 64bits to this		
          #elif OGRE_SIMD_V2_64_SSE2
            const __m128d a = simd;        // load a from this
            const __m128d b = other.simd;  // load b from other
            simd = b;                      // save b to this
            other.simd = a;                // save a to other
          #elif OGRE_SIMD_V2_64U_SSE2
            const __m128d a = _mm_loadu_pd(vals);        // load a from this
            const __m128d b = _mm_loadu_pd(other.vals);  // load b from other
            _mm_storeu_pd(vals, b);                      // save b to this
            _mm_storeu_pd(other.vals, a);                // save a to other
          #else
            std::swap(x, other.x);
            std::swap(y, other.y);
          #endif
        }

        inline Real operator [] ( const size_t i ) const
        {
            assert( i < 2 );

            return *(&x+i);
        }

        inline Real& operator [] ( const size_t i )
        {
            assert( i < 2 );

            return *(&x+i);
        }

        /// Pointer accessor for direct copying
        inline Real* ptr()
        {
            return &x;
        }
        /// Pointer accessor for direct copying
        inline const Real* ptr() const
        {
            return &x;
        }

        /** Assigns the value of the other vector.
            @param
                rkVector The other vector
        */
        inline Vector2& operator = ( const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            _mm_storel_epi64((__m128i*)vals, _mm_loadl_epi64((__m128i*)rkVector.vals));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = rkVector.simd;
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_loadu_pd(rkVector.vals));
          #else
            x = rkVector.x;
            y = rkVector.y;
          #endif
            return *this;
        }

        inline Vector2& operator = ( const Real fScalar)
        {
          #if OGRE_SIMD_V2_32_SSE2
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_set1_pd(fScalar);
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_set1_pd(fScalar));
          #else
            x = fScalar;
            y = fScalar;
          #endif
            return *this;
        }

        inline bool operator == ( const Vector2& rkVector ) const
        {
            return ( x == rkVector.x && y == rkVector.y );
        }

        inline bool operator != ( const Vector2& rkVector ) const
        {
            return ( x != rkVector.x || y != rkVector.y  );
        }

        // arithmetic operations
        inline Vector2 operator + ( const Vector2& rkVector ) const
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            return Vector2(_mm_add_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_add_pd(simd, rkVector.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_add_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
            return Vector2(
                x + rkVector.x,
                y + rkVector.y);
          #endif
        }

        inline Vector2 operator - ( const Vector2& rkVector ) const
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            return Vector2(_mm_sub_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_sub_pd(simd, rkVector.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_sub_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
            return Vector2(
                x - rkVector.x,
                y - rkVector.y);
          #endif
        }

        inline Vector2 operator * ( const Real fScalar ) const
        {
          #if OGRE_SIMD_V2_32_SSE2
            return Vector2(_mm_mul_ps(_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_mul_pd(simd, _mm_set1_pd(fScalar)));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_mul_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScalar)));
          #else
            return Vector2(
                x * fScalar,
                y * fScalar);
          #endif
        }

        inline Vector2 operator * ( const Vector2& rhs) const
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rhs.vals));
            return Vector2(_mm_mul_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_mul_pd(simd, rhs.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_mul_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rhs.vals)));
          #else
            return Vector2(
                x * rhs.x,
                y * rhs.y);
          #endif
        }

        inline Vector2 operator / ( const Real fScalar ) const
        {
          assert( fScalar != 0.0 );
 
          #if OGRE_SIMD_V2_32_SSE2
            return Vector2(_mm_div_ps(_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals)), _mm_set1_ps(fScalar)));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_div_pd(simd, _mm_set1_pd(fScalar)));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_div_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScalar)));
          #else
            const Real fInv = 1.0f / fScalar;

            return Vector2(
                x * fInv,
                y * fInv);
          #endif
        }

        inline Vector2 operator / ( const Vector2& rhs) const
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rhs.vals));
            return Vector2(_mm_div_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_div_pd(simd, rhs.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_div_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rhs.vals)));
          #else
            return Vector2(
                x / rhs.x,
                y / rhs.y);
          #endif
        }

        inline const Vector2& operator + () const
        {
            return *this;
        }

        inline Vector2 operator - () const
        {
          #if OGRE_SIMD_V2_32_SSE2
			return Vector2(_mm_sub_ps(_mm_setzero_ps(), _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals))));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_sub_pd(_mm_setzero_pd(), simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_sub_pd(_mm_setzero_pd(), _mm_loadu_pd(vals)));
          #else
            return Vector2(-x, -y);
          #endif
        }

        // overloaded operators to help Vector2
        inline friend Vector2 operator * ( const Real fScalar, const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_set1_ps(fScalar);
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));		
            return Vector2(_mm_mul_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_mul_pd(_mm_set1_pd(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_mul_pd(_mm_set1_pd(fScalar), _mm_loadu_pd(rkVector.vals)));
          #else
            return Vector2(
                fScalar * rkVector.x,
                fScalar * rkVector.y);
          #endif
        }

        inline friend Vector2 operator / ( const Real fScalar, const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_set1_ps(fScalar);
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            return Vector2(_mm_div_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_div_pd(_mm_set1_pd(fScalar), rkVector.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_div_pd(_mm_set1_pd(fScalar), _mm_loadu_pd(rkVector.vals)));
          #else
            return Vector2(
                fScalar / rkVector.x,
                fScalar / rkVector.y);
          #endif
        }

        inline friend Vector2 operator + (const Vector2& lhs, const Real rhs)
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)lhs.vals));
            const __m128 b = _mm_set1_ps(rhs);
            return Vector2(_mm_add_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_add_pd(lhs.simd, _mm_set1_pd(rhs)));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_add_pd(_mm_loadu_pd(lhs.vals), _mm_set1_pd(rhs)));
          #else
            return Vector2(
                lhs.x + rhs,
                lhs.y + rhs);
          #endif
        }

        inline friend Vector2 operator + (const Real lhs, const Vector2& rhs)
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_set1_ps(lhs);
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rhs.vals));
            return Vector2(_mm_add_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_add_pd(_mm_set1_pd(lhs), rhs.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_add_pd(_mm_set1_pd(lhs), _mm_loadu_pd(rhs.vals)));
          #else
            return Vector2(
                lhs + rhs.x,
                lhs + rhs.y);
          #endif
        }

        inline friend Vector2 operator - (const Vector2& lhs, const Real rhs)
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)lhs.vals));
            const __m128 b = _mm_set1_ps(rhs);
            return Vector2(_mm_sub_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_sub_pd(lhs.simd, _mm_set1_pd(rhs)));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_sub_pd(_mm_loadu_pd(lhs.vals), _mm_set1_pd(rhs)));
          #else
            return Vector2(
                lhs.x - rhs,
                lhs.y - rhs);
          #endif
        }

        inline friend Vector2 operator - (const Real lhs, const Vector2& rhs)
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_set1_ps(lhs);
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rhs.vals));
            return Vector2(_mm_sub_ps(a, b));
          #elif OGRE_SIMD_V2_64_SSE2
            return Vector2(_mm_sub_pd(_mm_set1_pd(lhs), rhs.simd));
          #elif OGRE_SIMD_V2_64U_SSE2
            return Vector2(_mm_sub_pd(_mm_set1_pd(lhs), _mm_loadu_pd(rhs.vals)));
          #else
            return Vector2(
                lhs - rhs.x,
                lhs - rhs.y);
          #endif
        }

        // arithmetic updates
        inline Vector2& operator += ( const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_add_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_add_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_add_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
            x += rkVector.x;
            y += rkVector.y;
          #endif
            return *this;
        }

        inline Vector2& operator += ( const Real fScaler )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_set1_ps(fScaler);
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_add_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_add_pd(simd, _mm_set1_pd(fScaler));
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_add_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScaler)));
          #else
            x += fScaler;
            y += fScaler;
          #endif
            return *this;
        }

        inline Vector2& operator -= ( const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_sub_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_sub_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_sub_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
            x -= rkVector.x;
            y -= rkVector.y;
          #endif
            return *this;
        }

        inline Vector2& operator -= ( const Real fScaler )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_set1_ps(fScaler);
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_sub_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_sub_pd(simd, _mm_set1_pd(fScaler));
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_sub_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScaler)));
          #else
            x -= fScaler;
            y -= fScaler;
          #endif
            return *this;
        }

        inline Vector2& operator *= ( const Real fScalar )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_set1_ps(fScalar);
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_mul_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_mul_pd(simd, _mm_set1_pd(fScalar));
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_mul_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScalar)));
          #else
            x *= fScalar;
            y *= fScalar;
          #endif
            return *this;
        }

        inline Vector2& operator *= ( const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_mul_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_mul_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_mul_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
            x *= rkVector.x;
            y *= rkVector.y;
          #endif
            return *this;
        }

        inline Vector2& operator /= ( const Real fScalar )
        {
          assert( fScalar != 0.0 );

          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_set1_ps(fScalar);
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_div_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_div_pd(simd, _mm_set1_pd(fScalar));
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_div_pd(_mm_loadu_pd(vals), _mm_set1_pd(fScalar)));
          #else
            const Real fInv = 1.0f / fScalar;

            x *= fInv;
            y *= fInv;
          #endif
            return *this;
        }

        inline Vector2& operator /= ( const Vector2& rkVector )
        {
          #if OGRE_SIMD_V2_32_SSE2
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)rkVector.vals));
            _mm_storel_epi64((__m128i*)vals, _mm_castps_si128(_mm_div_ps(a, b)));
          #elif OGRE_SIMD_V2_64_SSE2
            simd = _mm_div_pd(simd, rkVector.simd);
          #elif OGRE_SIMD_V2_64U_SSE2
            _mm_storeu_pd(vals, _mm_div_pd(_mm_loadu_pd(vals), _mm_loadu_pd(rkVector.vals)));
          #else
			x /= rkVector.x;
            y /= rkVector.y;
          #endif
            return *this;
        }

        /** Returns the length (magnitude) of the vector.
            @warning
                This operation requires a square root and is expensive in
                terms of CPU operations. If you don't need to know the exact
                length (e.g. for just comparing lengths) use squaredLength()
                instead.
        */
        inline Real length () const
        {
          #if OGRE_SIMD_V2_32_SSE41
            const __m128 a = _mm_castsi128_ps(_mm_loadl_epi64((__m128i*)vals));
            const __m128 b = _mm_dp_ps(a, a, 0x31);
            const __m128 c = _mm_sqrt_ss(b);
            return c.m128_f32[0];
          #elif OGRE_SIMD_V2_64_SSE41
            const __m128d b = _mm_dp_pd(simd, simd, 0x31);
            const __m128d c = _mm_sqrt_sd(b, b);
            return c.m128d_f64[0];
          #elif OGRE_SIMD_V2_64U_SSE41
            const __m128d a = _mm_loadu_pd(vals);
            const __m128d b = _mm_dp_pd(a, a, 0x31);
            const __m128d c = _mm_sqrt_sd(b, b);
            return c.m128d_f64[0];
          #else
            return Math::Sqrt( x * x + y * y );
          #endif
        }

        /** Returns the square of the length(magnitude) of the vector.
            @remarks
                This  method is for efficiency - calculating the actual
                length of a vector requires a square root, which is expensive
                in terms of the operations required. This method returns the
                square of the length of the vector, i.e. the same as the
                length but before the square root is taken. Use this if you
                want to find the longest / shortest vector without incurring
                the square root.
        */
        inline Real squaredLength () const
        {
          #if OGRE_SIMD_V2_64_SSE41
            const __m128d b = _mm_dp_pd(simd, simd, 0x31);
            return b.m128d_f64[0];
          #elif OGRE_SIMD_V2_64U_SSE41
            const __m128d a = _mm_loadu_pd(vals);
            const __m128d b = _mm_dp_pd(a, a, 0x31);
            return b.m128d_f64[0];
          #else
            return x * x + y * y;
          #endif
        }

        /** Returns the distance to another vector.
            @warning
                This operation requires a square root and is expensive in
                terms of CPU operations. If you don't need to know the exact
                distance (e.g. for just comparing distances) use squaredDistance()
                instead.
        */
        inline Real distance(const Vector2& rhs) const
        {
          #if OGRE_SIMD_V2_64_SSE41
            const __m128d c = _mm_sub_pd(simd, rhs.simd);
            const __m128d d = _mm_dp_pd(c, c, 0x31);
            const __m128d e = _mm_sqrt_sd(d, d);
            return e.m128d_f64[0];
          #elif OGRE_SIMD_V2_64U_SSE41
            const __m128d a = _mm_loadu_pd(vals);
            const __m128d b = _mm_loadu_pd(rhs.vals);
            const __m128d c = _mm_sub_pd(a, b);
            const __m128d d = _mm_dp_pd(c, c, 0x31);
            const __m128d e = _mm_sqrt_sd(d, d);
            return e.m128d_f64[0];
          #else
            return (*this - rhs).length();
          #endif
        }

        /** Returns the square of the distance to another vector.
            @remarks
                This method is for efficiency - calculating the actual
                distance to another vector requires a square root, which is
                expensive in terms of the operations required. This method
                returns the square of the distance to another vector, i.e.
                the same as the distance but before the square root is taken.
                Use this if you want to find the longest / shortest distance
                without incurring the square root.
        */
        inline Real squaredDistance(const Vector2& rhs) const
        {
          #if OGRE_SIMD_V2_64_SSE41
            const __m128d c = _mm_sub_pd(simd, rhs.simd);
            const __m128d d = _mm_dp_pd(c, c, 0x31);
            return d.m128d_f64[0];
          #elif OGRE_SIMD_V2_64U_SSE41
            const __m128d a = _mm_loadu_pd(vals);
            const __m128d b = _mm_loadu_pd(rhs.vals);
            const __m128d c = _mm_sub_pd(a, b);
            const __m128d d = _mm_dp_pd(c, c, 0x31);
            return d.m128d_f64[0];
          #else
            return (*this - rhs).squaredLength();
          #endif
        }

        /** Calculates the dot (scalar) product of this vector with another.
            @remarks
                The dot product can be used to calculate the angle between 2
                vectors. If both are unit vectors, the dot product is the
                cosine of the angle; otherwise the dot product must be
                divided by the product of the lengths of both vectors to get
                the cosine of the angle. This result can further be used to
                calculate the distance of a point from a plane.
            @param
                vec Vector with which to calculate the dot product (together
                with this one).
            @return
                A float representing the dot product value.
        */
        inline Real dotProduct(const Vector2& vec) const
        {
          #if OGRE_SIMD_V2_64_SSE41
            return _mm_dp_pd(simd, vec.simd, 0x31).m128d_f64[0];
          #elif OGRE_SIMD_V2_64U_SSE41
            const __m128d a = _mm_loadu_pd(vals);
            const __m128d b = _mm_loadu_pd(vec.vals);
            return _mm_dp_pd(a, b, 0x31).m128d_f64[0];
          #else
            return x * vec.x + y * vec.y;
          #endif
        }

        /** Normalises the vector.
            @remarks
                This method normalises the vector such that it's
                length / magnitude is 1. The result is called a unit vector.
            @note
                This function will not crash for zero-sized vectors, but there
                will be no changes made to their components.
            @return The previous length of the vector.
        */

        inline Real normalise()
        {
            Real fLength = length();

            // Will also work for zero-sized vectors, but will change nothing
            // We're not using epsilons because we don't need to.
            // Read http://www.ogre3d.org/forums/viewtopic.php?f=4&t=61259
            if ( fLength > Real(0.0f) )
            {
                Real fInvLength = 1.0f / fLength;
                x *= fInvLength;
                y *= fInvLength;
            }

            return fLength;
        }

        /** Returns a vector at a point half way between this and the passed
            in vector.
        */
        inline Vector2 midPoint( const Vector2& vec ) const
        {
          #if OGRE_SIMD_V2_64_SSE2
            const __m128d a = _mm_add_pd(simd, vec.simd);
            const __m128d b = _mm_mul_pd(a, _mm_set1_pd(0.5));
            return Vector2(b);
          #elif OGRE_SIMD_V2_64U_SSE2
            const __m128d a = _mm_add_pd(_mm_loadu_pd(vals), _mm_loadu_pd(vec.vals));
            const __m128d b = _mm_mul_pd(a, _mm_set1_pd(0.5));
            return Vector2(b);
          #else
            return Vector2(
                ( x + vec.x ) * (Real)0.5f,
                ( y + vec.y ) * (Real)0.5f );
          #endif
        }

        /** Returns true if the vector's scalar components are all greater
            that the ones of the vector it is compared against.
        */
        inline bool operator < ( const Vector2& rhs ) const
        {
            if( x < rhs.x && y < rhs.y )
                return true;
            return false;
        }

        /** Returns true if the vector's scalar components are all smaller
            that the ones of the vector it is compared against.
        */
        inline bool operator > ( const Vector2& rhs ) const
        {
            if( x > rhs.x && y > rhs.y )
                return true;
            return false;
        }

        /** Sets this vector's components to the minimum of its own and the
            ones of the passed in vector.
            @remarks
                'Minimum' in this case means the combination of the lowest
                value of x, y and z from both vectors. Lowest is taken just
                numerically, not magnitude, so -1 < 0.
        */
        inline void makeFloor( const Vector2& cmp )
        {
            if( cmp.x < x ) x = cmp.x;
            if( cmp.y < y ) y = cmp.y;
        }

        /** Sets this vector's components to the maximum of its own and the
            ones of the passed in vector.
            @remarks
                'Maximum' in this case means the combination of the highest
                value of x, y and z from both vectors. Highest is taken just
                numerically, not magnitude, so 1 > -3.
        */
        inline void makeCeil( const Vector2& cmp )
        {
            if( cmp.x > x ) x = cmp.x;
            if( cmp.y > y ) y = cmp.y;
        }

        /** Generates a vector perpendicular to this vector (eg an 'up' vector).
            @remarks
                This method will return a vector which is perpendicular to this
                vector. There are an infinite number of possibilities but this
                method will guarantee to generate one of them. If you need more
                control you should use the Quaternion class.
        */
        inline Vector2 perpendicular(void) const
        {
            return Vector2(-y, x);
        }

        /** Calculates the 2 dimensional cross-product of 2 vectors, which results
            in a single floating point value which is 2 times the area of the triangle.
        */
        inline Real crossProduct( const Vector2& rkVector ) const
        {
            return x * rkVector.y - y * rkVector.x;
        }

        /** Generates a new random vector which deviates from this vector by a
            given angle in a random direction.
            @remarks
                This method assumes that the random number generator has already
                been seeded appropriately.
            @param angle
                The angle at which to deviate in radians
            @return
                A random vector which deviates from this vector by angle. This
                vector will not be normalised, normalise it if you wish
                afterwards.
        */
        inline Vector2 randomDeviant(Radian angle) const
        {
            angle *= Math::RangeRandom(-1, 1);
            Real cosa = Math::Cos(angle);
            Real sina = Math::Sin(angle);
            return Vector2(cosa * x - sina * y,
                               sina * x + cosa * y);
        }

        /** Returns true if this vector is zero length. */
        inline bool isZeroLength(void) const
        {
            Real sqlen = (x * x) + (y * y);
            return (sqlen < (1e-06 * 1e-06));

        }

        /** As normalise, except that this vector is unaffected and the
            normalised vector is returned as a copy. */
        inline Vector2 normalisedCopy(void) const
        {
			Vector2 ret = *this;
            ret.normalise();
            return ret;
        }

        /** Calculates a reflection vector to the plane with the given normal .
        @remarks NB assumes 'this' is pointing AWAY FROM the plane, invert if it is not.
        */
        inline Vector2 reflect(const Vector2& normal) const
        {
            return Vector2( *this - ( 2 * this->dotProduct(normal) * normal ) );
        }

        /// Check whether this vector contains valid values
        inline bool isNaN() const
        {
            return Math::isNaN(x) || Math::isNaN(y);
        }

        /**  Gets the angle between 2 vectors.
        @remarks
            Vectors do not have to be unit-length but must represent directions.
        */
        inline Ogre::Radian angleBetween(const Ogre::Vector2& other) const
        {       
            Ogre::Real lenProduct = length() * other.length();
            // Divide by zero check
            if(lenProduct < 1e-6f)
                lenProduct = 1e-6f;
        
            Ogre::Real f = dotProduct(other) / lenProduct;
    
            f = Ogre::Math::Clamp(f, (Ogre::Real)-1.0, (Ogre::Real)1.0);
            return Ogre::Math::ACos(f);
        }

        /**  Gets the oriented angle between 2 vectors.
        @remarks
            Vectors do not have to be unit-length but must represent directions.
            The angle is comprised between 0 and 2 PI.
        */
        inline Ogre::Radian angleTo(const Ogre::Vector2& other) const
        {
            Ogre::Radian angle = angleBetween(other);
        
            if (crossProduct(other)<0)          
                angle = (Ogre::Radian)Ogre::Math::TWO_PI - angle;       

            return angle;
        }

        // special points
        static const Vector2 ZERO;
        static const Vector2 UNIT_X;
        static const Vector2 UNIT_Y;
        static const Vector2 NEGATIVE_UNIT_X;
        static const Vector2 NEGATIVE_UNIT_Y;
        static const Vector2 UNIT_SCALE;

        /** Function for writing to a stream.
        */
        inline _OgreExport friend std::ostream& operator <<
            ( std::ostream& o, const Vector2& v )
        {
            o << "Vector2(" << v.x << ", " << v.y <<  ")";
            return o;
        }
    };
    
	/** @} */
    /** @} */

}
#endif
