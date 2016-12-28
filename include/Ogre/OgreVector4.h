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
	class _OgreExport Vector4Base
    {
    public:
        Real x, y, z, w;

    public:
        /** Default constructor.
            @note
                It does <b>NOT</b> initialize the vector for efficiency.
        */
        inline Vector4Base()
        {
        }

        inline Vector4Base( const Real fX, const Real fY, const Real fZ, const Real fW )
            : x( fX ), y( fY ), z( fZ ), w( fW )
        {
        }

        inline explicit Vector4Base( const Real afCoordinate[4] )
            : x( afCoordinate[0] ),
              y( afCoordinate[1] ),
              z( afCoordinate[2] ),
              w( afCoordinate[3] )
        {
        }

        inline explicit Vector4Base( const int afCoordinate[4] )
        {
			x = (Real)afCoordinate[0];
			y = (Real)afCoordinate[1];
			z = (Real)afCoordinate[2];
			w = (Real)afCoordinate[3];
        }

        inline explicit Vector4Base( Real* const r )
            : x( r[0] ), y( r[1] ), z( r[2] ), w( r[3] )
        {
        }

        inline explicit Vector4Base(const Real scaler)
            : x(scaler)
            , y(scaler)
            , z(scaler)
            , w(scaler)
        {
        }

        inline explicit Vector4Base(const Vector3& rhs)
            : x(rhs.x), y(rhs.y), z(rhs.z), w(1.0f)
        {
        }

        /** Swizzle-like narrowing operations
        */
        inline Vector3 xyz() const
        {
            return Vector3(x, y, z);
        }
        inline Vector2 xy() const
        {
            return Vector2(x, y);
        }

        /** Exchange the contents of this vector with another. 
        */
        inline void swap(Vector4Base& other)
        {
            std::swap(x, other.x);
            std::swap(y, other.y);
            std::swap(z, other.z);
            std::swap(w, other.w);
        }
    
        inline Real operator [] ( const size_t i ) const
        {
            assert( i < 4 );

            return *(&x+i);
        }

        inline Real& operator [] ( const size_t i )
        {
            assert( i < 4 );

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
        inline Vector4Base& operator = ( const Vector4Base& rkVector )
        {
            x = rkVector.x;
            y = rkVector.y;
            z = rkVector.z;
            w = rkVector.w;

            return *this;
        }

        inline Vector4Base& operator = ( const Real fScalar)
        {
            x = fScalar;
            y = fScalar;
            z = fScalar;
            w = fScalar;
            return *this;
        }

        inline bool operator == ( const Vector4Base& rkVector ) const
        {
            return ( x == rkVector.x &&
                y == rkVector.y &&
                z == rkVector.z &&
                w == rkVector.w );
        }

        inline bool operator != ( const Vector4Base& rkVector ) const
        {
            return ( x != rkVector.x ||
                y != rkVector.y ||
                z != rkVector.z ||
                w != rkVector.w );
        }

        inline Vector4Base& operator = (const Vector3& rhs)
        {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
            w = 1.0f;
            return *this;
        }

        // arithmetic operations
        inline Vector4Base operator + ( const Vector4Base& rkVector ) const
        {
            return Vector4Base(
                x + rkVector.x,
                y + rkVector.y,
                z + rkVector.z,
                w + rkVector.w);
        }

        inline Vector4Base operator - ( const Vector4Base& rkVector ) const
        {
            return Vector4Base(
                x - rkVector.x,
                y - rkVector.y,
                z - rkVector.z,
                w - rkVector.w);
        }

        inline Vector4Base operator * ( const Real fScalar ) const
        {
            return Vector4Base(
                x * fScalar,
                y * fScalar,
                z * fScalar,
                w * fScalar);
        }

        inline Vector4Base operator * ( const Vector4Base& rhs) const
        {
            return Vector4Base(
                rhs.x * x,
                rhs.y * y,
                rhs.z * z,
                rhs.w * w);
        }

        inline Vector4Base operator / ( const Real fScalar ) const
        {
            assert( fScalar != 0.0 );

            Real fInv = 1.0f / fScalar;

            return Vector4Base(
                x * fInv,
                y * fInv,
                z * fInv,
                w * fInv);
        }

        inline Vector4Base operator / ( const Vector4Base& rhs) const
        {
            return Vector4Base(
                x / rhs.x,
                y / rhs.y,
                z / rhs.z,
                w / rhs.w);
        }

        inline const Vector4Base& operator + () const
        {
            return *this;
        }

        inline Vector4Base operator - () const
        {
            return Vector4Base(-x, -y, -z, -w);
        }

        inline friend Vector4Base operator * ( const Real fScalar, const Vector4Base& rkVector )
        {
            return Vector4Base(
                fScalar * rkVector.x,
                fScalar * rkVector.y,
                fScalar * rkVector.z,
                fScalar * rkVector.w);
        }

        inline friend Vector4Base operator / ( const Real fScalar, const Vector4Base& rkVector )
        {
            return Vector4Base(
                fScalar / rkVector.x,
                fScalar / rkVector.y,
                fScalar / rkVector.z,
                fScalar / rkVector.w);
        }

        inline friend Vector4Base operator + (const Vector4Base& lhs, const Real rhs)
        {
            return Vector4Base(
                lhs.x + rhs,
                lhs.y + rhs,
                lhs.z + rhs,
                lhs.w + rhs);
        }

        inline friend Vector4Base operator + (const Real lhs, const Vector4Base& rhs)
        {
            return Vector4Base(
                lhs + rhs.x,
                lhs + rhs.y,
                lhs + rhs.z,
                lhs + rhs.w);
        }

        inline friend Vector4Base operator - (const Vector4Base& lhs, Real rhs)
        {
            return Vector4Base(
                lhs.x - rhs,
                lhs.y - rhs,
                lhs.z - rhs,
                lhs.w - rhs);
        }

        inline friend Vector4Base operator - (const Real lhs, const Vector4Base& rhs)
        {
            return Vector4Base(
                lhs - rhs.x,
                lhs - rhs.y,
                lhs - rhs.z,
                lhs - rhs.w);
        }

        // arithmetic updates
        inline Vector4Base& operator += ( const Vector4Base& rkVector )
        {
            x += rkVector.x;
            y += rkVector.y;
            z += rkVector.z;
            w += rkVector.w;

            return *this;
        }

        inline Vector4Base& operator -= ( const Vector4Base& rkVector )
        {
            x -= rkVector.x;
            y -= rkVector.y;
            z -= rkVector.z;
            w -= rkVector.w;

            return *this;
        }

        inline Vector4Base& operator *= ( const Real fScalar )
        {
            x *= fScalar;
            y *= fScalar;
            z *= fScalar;
            w *= fScalar;

			return *this;
        }

        inline Vector4Base& operator += ( const Real fScalar )
        {
            x += fScalar;
            y += fScalar;
            z += fScalar;
            w += fScalar;

            return *this;
        }

        inline Vector4Base& operator -= ( const Real fScalar )
        {
            x -= fScalar;
            y -= fScalar;
            z -= fScalar;
            w -= fScalar;

            return *this;
        }

        inline Vector4Base& operator *= ( const Vector4Base& rkVector )
        {
            x *= rkVector.x;
            y *= rkVector.y;
            z *= rkVector.z;
            w *= rkVector.w;

            return *this;
        }

        inline Vector4Base& operator /= ( const Real fScalar )
        {
            assert( fScalar != 0.0 );

            Real fInv = 1.0f / fScalar;

            x *= fInv;
            y *= fInv;
            z *= fInv;
            w *= fInv;

            return *this;
        }

        inline Vector4Base& operator /= ( const Vector4Base& rkVector )
        {
            x /= rkVector.x;
            y /= rkVector.y;
            z /= rkVector.z;
            w /= rkVector.w;

            return *this;
        }

        /** Calculates the dot (scalar) product of this vector with another.
            @param
                vec Vector with which to calculate the dot product (together
                with this one).
            @return
                A float representing the dot product value.
        */
        inline Real dotProduct(const Vector4Base& vec) const
        {
            return x * vec.x + y * vec.y + z * vec.z + w * vec.w;
        }
        /// Check whether this vector contains valid values
        inline bool isNaN() const
        {
            return Math::isNaN(x) || Math::isNaN(y) || Math::isNaN(z) || Math::isNaN(w);
        }
        /** Function for writing to a stream.
        */
        inline _OgreExport friend std::ostream& operator <<
            ( std::ostream& o, const Vector4Base& v )
        {
            o << "Vector4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
            return o;
        }
        // special
        static const Vector4Base ZERO;
    };

#if OGRE_SIMD_SSE_SINGLE
	static const __m128 SIGNMASK_SSE = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

	/** \addtogroup Core
	*  @{
	*/
	/** \addtogroup Math
	*  @{
	*/
	/** 4-dimensional homogeneous single precision vector using SSE.
	*/
	ALIGN16 class _OgreExport Vector4SSE32 : public Vector4Base
	{
	public:
		FORCEINLINE Vector4SSE32()
		{
		}

		FORCEINLINE Vector4SSE32(const Real fX, const Real fY, const Real fZ, const Real fW)
		{
			_mm_store_ps(ptr(), _mm_set_ps(fX, fY, fZ, fW));
		}

		FORCEINLINE explicit Vector4SSE32(const Real afCoordinate[4])
		{
			_mm_store_ps(ptr(), _mm_loadu_ps(afCoordinate));
		}

		FORCEINLINE explicit Vector4SSE32(const int afCoordinate[4])
		{
			const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
			const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
			const __m128  c = _mm_cvtepi32_ps(b);                       // convert ints to floats
			_mm_store_ps(ptr(), c);
		}

		FORCEINLINE explicit Vector4SSE32(Real* const r)
		{
			_mm_store_ps(ptr(), _mm_loadu_ps(r));
		}

		FORCEINLINE explicit Vector4SSE32(const Real scaler)
		{
			_mm_store_ps(ptr(), _mm_set1_ps(scaler));
		}

		FORCEINLINE explicit Vector4SSE32(const Vector3& rhs)
			: Vector4Base(rhs)
		{
		}

		// Copy Constructor
		FORCEINLINE Vector4SSE32(const Vector4SSE32& other)
		{
			_mm_store_ps(ptr(), _mm_load_ps(other.ptr()));
		}

		// SSE Constructor
		FORCEINLINE Vector4SSE32(const __m128 values)
		{
			_mm_store_ps(ptr(), values);
		}

		/** Exchange the contents of this vector with another.
		*/
		FORCEINLINE void swap(Vector4SSE32& other)
		{
			const __m128 a = _mm_load_ps(ptr());        // load a from this
			const __m128 b = _mm_load_ps(other.ptr());  // load b from other
			_mm_store_ps(ptr(), b);                     // save b to this
			_mm_store_ps(other.ptr(), a);               // save a to other
		}

		/** Assigns the value of the other vector.
		@param
		rkVector The other vector
		*/
		FORCEINLINE Vector4SSE32& operator = (const Vector4SSE32& rkVector)
		{
			_mm_store_ps(ptr(), _mm_load_ps(rkVector.ptr()));
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator = (const Real fScalar)
		{
			_mm_store_ps(ptr(), _mm_set1_ps(fScalar));
			return *this;
		}

		FORCEINLINE bool operator == (const Vector4SSE32& rkVector) const
		{
			const __m128  a = _mm_load_ps(ptr());          // load this
			const __m128  b = _mm_load_ps(rkVector.ptr()); // load other
			const __m128  c = _mm_cmpeq_ps(a, b);          // compare all components
			const __m128i d = _mm_castps_si128(c);         // cast from float to int (no-op)
			const int     e = _mm_movemask_epi8(d);        // combine some bits from all registers
			const bool    f = e == 0xFFFF;                 // check all equal
			return f;
		}

		FORCEINLINE bool operator != (const Vector4SSE32& rkVector) const
		{
			const __m128  a = _mm_load_ps(ptr());          // load this
			const __m128  b = _mm_load_ps(rkVector.ptr()); // load other
			const __m128  c = _mm_cmpeq_ps(a, b);          // compare all components
			const __m128i d = _mm_castps_si128(c);         // cast from float to int (no-op)
			const int     e = _mm_movemask_epi8(d);        // combine some bits from all registers
			const bool    f = e != 0xFFFF;                 // check any not equal
			return f;
		}

		FORCEINLINE Vector4SSE32& operator = (const Vector3& rhs)
		{
			x = rhs.x;
			y = rhs.y;
			z = rhs.z;
			w = 1.0f;
			return *this;
		}

		// arithmetic operations
		FORCEINLINE Vector4SSE32 operator + (const Vector4SSE32& rkVector) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_add_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE Vector4SSE32 operator - (const Vector4SSE32& rkVector) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_sub_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE Vector4SSE32 operator * (const Real fScalar) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_mul_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE Vector4SSE32 operator * (const Vector4SSE32& rhs) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rhs.ptr());
			const __m128 c = _mm_mul_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE Vector4SSE32 operator / (const Real fScalar) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_div_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE Vector4SSE32 operator / (const Vector4SSE32& rhs) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rhs.ptr());
			const __m128 c = _mm_div_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE const Vector4SSE32& operator + () const
		{
			return *this;
		}

		FORCEINLINE Vector4SSE32 operator - () const
		{
			const __m128 a = _mm_load_ps(ptr());          // load values
			const __m128 b = _mm_xor_ps(a, SIGNMASK_SSE); // flip sign bit
			return Vector4SSE32(b);
		}

		FORCEINLINE friend Vector4SSE32 operator * (const Real fScalar, const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_set1_ps(fScalar);
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_mul_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE friend Vector4SSE32 operator / (const Real fScalar, const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_set1_ps(fScalar);
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_div_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE friend Vector4SSE32 operator + (const Vector4SSE32& lhs, const Real rhs)
		{
			const __m128 a = _mm_load_ps(lhs.ptr());
			const __m128 b = _mm_set1_ps(rhs);
			const __m128 c = _mm_add_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE friend Vector4SSE32 operator + (const Real lhs, const Vector4SSE32& rhs)
		{
			const __m128 a = _mm_set1_ps(lhs);
			const __m128 b = _mm_load_ps(rhs.ptr());
			const __m128 c = _mm_add_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE friend Vector4SSE32 operator - (const Vector4SSE32& lhs, Real rhs)
		{
			const __m128 a = _mm_load_ps(lhs.ptr());
			const __m128 b = _mm_set1_ps(rhs);
			const __m128 c = _mm_sub_ps(a, b);
			return Vector4SSE32(c);
		}

		FORCEINLINE friend Vector4SSE32 operator - (const Real lhs, const Vector4SSE32& rhs)
		{
			const __m128 a = _mm_set1_ps(lhs);
			const __m128 b = _mm_load_ps(rhs.ptr());
			const __m128 c = _mm_sub_ps(a, b);
			return Vector4SSE32(c);
		}

		// arithmetic updates
		FORCEINLINE Vector4SSE32& operator += (const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_add_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator -= (const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_sub_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator *= (const Real fScalar)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_mul_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator += (const Real fScalar)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_add_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator -= (const Real fScalar)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_sub_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator *= (const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_mul_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator /= (const Real fScalar)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_set1_ps(fScalar);
			const __m128 c = _mm_div_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

		FORCEINLINE Vector4SSE32& operator /= (const Vector4SSE32& rkVector)
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(rkVector.ptr());
			const __m128 c = _mm_div_ps(a, b);
			_mm_store_ps(ptr(), c);
			return *this;
		}

#if OGRE_SIMD_TYPES_SSE41 == 1
		FORCEINLINE Real dotProduct(const Vector4SSE32& vec) const
		{
			const __m128 a = _mm_load_ps(ptr());
			const __m128 b = _mm_load_ps(vec.ptr());
			const __m128 c = _mm_dp_ps(a, b, 0xFF);
			return c.m128_f32[0];
		}
#endif

		/// Check whether this vector contains valid values
		FORCEINLINE bool isNaN() const
		{
			const __m128  a = _mm_load_ps(ptr());
			const __m128  b = _mm_cmpeq_ps(a, a);          // NaN has: (NaN == NaN) = false
			const __m128i c = _mm_castps_si128(b);         // cast from float to int (no-op)
			const int     d = _mm_movemask_epi8(c);        // combine some bits from all registers
			const bool    e = d != 0xFFFF;                 // check not all equal
			return e;
		}

		// special
		static const Vector4SSE32 ZERO;
	};
#endif

#if OGRE_SIMD_AVX_DOUBLE
	static const __m256d SIGNMASK_AVX = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));

	/** \addtogroup Core
	*  @{
	*/
	/** \addtogroup Math
	*  @{
	*/
	/** 4-dimensional homogeneous double precision vector using AVX.
	*/
	ALIGN32 class _OgreExport Vector4AVX64 : public Vector4Base
	{
	public:
		FORCEINLINE Vector4AVX64() : Vector4Base()
		{
		}

		FORCEINLINE Vector4AVX64(const Real fX, const Real fY, const Real fZ, const Real fW)
		{
			_mm256_store_pd(ptr(), _mm256_set_pd(fX, fY, fZ, fW));
		}

		FORCEINLINE explicit Vector4AVX64(const Real afCoordinate[4])
		{
			_mm256_store_pd(ptr(), _mm256_loadu_pd(afCoordinate));
		}

		FORCEINLINE explicit Vector4AVX64(const int afCoordinate[4])
		{
			const __m128i a = _mm_set1_epi32(0xFFFFFFFF);               // set mask to load all 4 ints
			const __m128i b = _mm_maskload_epi32(&afCoordinate[0], a);  // load the 4 ints
			const __m256d c = _mm256_cvtepi32_pd(b);                    // convert ints to floats
			_mm256_store_pd(ptr(), c);
		}

		FORCEINLINE explicit Vector4AVX64(Real* const r)
		{
			_mm256_store_pd(ptr(), _mm256_loadu_pd(r));
		}

		FORCEINLINE explicit Vector4AVX64(const Real scaler)
		{
			_mm256_store_pd(ptr(), _mm256_set1_pd(scaler));
		}

		FORCEINLINE explicit Vector4AVX64(const Vector3& rhs)
			: Vector4Base(rhs)
		{
		}

		// Copy Constructor
		FORCEINLINE Vector4AVX64(const Vector4AVX64& other)
		{
			_mm256_store_pd(ptr(), _mm256_load_pd(other.ptr()));
		}

		// AVX Constructor
		FORCEINLINE Vector4AVX64(const __m256d values)
		{
			_mm256_store_pd(ptr(), values);
		}

		/** Exchange the contents of this vector with another.
		*/
		FORCEINLINE void swap(Vector4AVX64& other)
		{
			const __m256d a = _mm256_load_pd(ptr());        // load a from this
			const __m256d b = _mm256_load_pd(other.ptr());  // load b from other
			_mm256_store_pd(ptr(), b);                      // save b to this
			_mm256_store_pd(other.ptr(), a);                // save a to other
		}

		/** Assigns the value of the other vector.
		@param
		rkVector The other vector
		*/
		inline Vector4AVX64& operator = (const Vector4AVX64& rkVector)
		{
			_mm256_store_pd(ptr(), _mm256_load_pd(rkVector.ptr()));
			return *this;
		}

		inline Vector4AVX64& operator = (const Real fScalar)
		{
			_mm256_store_pd(ptr(), _mm256_set1_pd(fScalar));
			return *this;
		}

		inline bool operator == (const Vector4AVX64& rkVector) const
		{
			const __m256d a = _mm256_load_pd(ptr());           // load this
			const __m256d b = _mm256_load_pd(rkVector.ptr());  // load other
			const __m256d c = _mm256_cmp_pd(a, b, _CMP_EQ_OQ); // compare all components			
			const __m256i d = _mm256_castpd_si256(c);          // cast from double to int64 (no-op)
			const int     e = _mm256_movemask_epi8(d);         // combine some bits from all registers
			const bool    f = e == 0xFFFFFFFF;                 // check all equal
			return f;
		}

		inline bool operator != (const Vector4AVX64& rkVector) const
		{
			const __m256d a = _mm256_load_pd(ptr());           // load this
			const __m256d b = _mm256_load_pd(rkVector.ptr());  // load other
			const __m256d c = _mm256_cmp_pd(a, b, _CMP_EQ_OQ); // compare all components			
			const __m256i d = _mm256_castpd_si256(c);          // cast from double to int64 (no-op)
			const int     e = _mm256_movemask_epi8(d);         // combine some bits from all registers
			const bool    f = e != 0xFFFFFFFF;                 // check all equal
			return f;
		}

		inline Vector4AVX64& operator = (const Vector3& rhs)
		{
			x = rhs.x;
			y = rhs.y;
			z = rhs.z;
			w = 1.0f;
			return *this;
		}

		// arithmetic operations
		inline Vector4AVX64 operator + (const Vector4AVX64& rkVector) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_add_pd(a, b);
			return Vector4AVX64(c);
		}

		inline Vector4AVX64 operator - (const Vector4AVX64& rkVector) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_sub_pd(a, b);
			return Vector4AVX64(c);
		}

		inline Vector4AVX64 operator * (const Real fScalar) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_mul_pd(a, b);
			return Vector4AVX64(c);
		}

		inline Vector4AVX64 operator * (const Vector4AVX64& rhs) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rhs.ptr());
			const __m256d c = _mm256_mul_pd(a, b);
			return Vector4AVX64(c);
		}

		inline Vector4AVX64 operator / (const Real fScalar) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_div_pd(a, b);
			return Vector4AVX64(c);
		}

		inline Vector4AVX64 operator / (const Vector4AVX64& rhs) const
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rhs.ptr());
			const __m256d c = _mm256_div_pd(a, b);
			return Vector4AVX64(c);
		}

		inline const Vector4AVX64& operator + () const
		{
			return *this;
		}

		inline Vector4AVX64 operator - () const
		{
			const __m256d a = _mm256_load_pd(ptr());          // load values
			const __m256d b = _mm256_xor_pd(a, SIGNMASK_AVX); // flip sign bit
			return Vector4AVX64(b);
		}

		inline friend Vector4AVX64 operator * (const Real fScalar, const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_set1_pd(fScalar);
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_mul_pd(a, b);
			return Vector4AVX64(c);
		}

		inline friend Vector4AVX64 operator / (const Real fScalar, const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_set1_pd(fScalar);
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_div_pd(a, b);
			return Vector4AVX64(c);
		}

		inline friend Vector4AVX64 operator + (const Vector4AVX64& lhs, const Real rhs)
		{
			const __m256d a = _mm256_load_pd(lhs.ptr());
			const __m256d b = _mm256_set1_pd(rhs);
			const __m256d c = _mm256_add_pd(a, b);
			return Vector4AVX64(c);
		}

		inline friend Vector4AVX64 operator + (const Real lhs, const Vector4AVX64& rhs)
		{
			const __m256d a = _mm256_set1_pd(lhs);
			const __m256d b = _mm256_load_pd(rhs.ptr());
			const __m256d c = _mm256_add_pd(a, b);
			return Vector4AVX64(c);
		}

		inline friend Vector4AVX64 operator - (const Vector4AVX64& lhs, Real rhs)
		{
			const __m256d a = _mm256_load_pd(lhs.ptr());
			const __m256d b = _mm256_set1_pd(rhs);
			const __m256d c = _mm256_sub_pd(a, b);
			return Vector4AVX64(c);
		}

		inline friend Vector4AVX64 operator - (const Real lhs, const Vector4AVX64& rhs)
		{
			const __m256d a = _mm256_set1_pd(lhs);
			const __m256d b = _mm256_load_pd(rhs.ptr());
			const __m256d c = _mm256_sub_pd(a, b);
			return Vector4AVX64(c);
		}

		// arithmetic updates
		inline Vector4AVX64& operator += (const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_add_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator -= (const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_sub_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator *= (const Real fScalar)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_mul_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator += (const Real fScalar)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_add_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator -= (const Real fScalar)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_sub_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator *= (const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_mul_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}


		inline Vector4AVX64& operator /= (const Real fScalar)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_set1_pd(fScalar);
			const __m256d c = _mm256_div_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		inline Vector4AVX64& operator /= (const Vector4AVX64& rkVector)
		{
			const __m256d a = _mm256_load_pd(ptr());
			const __m256d b = _mm256_load_pd(rkVector.ptr());
			const __m256d c = _mm256_div_pd(a, b);
			_mm256_store_pd(ptr(), c);
			return *this;
		}

		// special
		static const Vector4AVX64 ZERO;
	};
#endif
    /** @} */
    /** @} */
}
#endif

