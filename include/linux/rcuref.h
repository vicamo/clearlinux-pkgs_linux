/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef _LINUX_RCUREF_H
#define _LINUX_RCUREF_H

#include <linux/atomic.h>
#include <linux/bug.h>
#include <linux/limits.h>
#include <linux/lockdep.h>
#include <linux/preempt.h>
#include <linux/rcupdate.h>

#define RCUREF_NOREF		0x00000000
#define RCUREF_ONEREF		0x00000001
#define RCUREF_MAXREF		0x7FFFFFFF
#define RCUREF_SATURATED	0xA0000000
#define RCUREF_RELEASED		0xC0000000
#define RCUREF_DEAD		0xE0000000

/**
 * rcuref_init - Initialize a rcuref reference count with the given reference count
 * @ref:	Pointer to the reference count
 * @cnt:	The initial reference count typically '1'
 */
static inline void rcuref_init(rcuref_t *ref, unsigned int cnt)
{
	atomic_set(&ref->refcnt, cnt);
}

/**
 * rcuref_read - Read the number of held reference counts of a rcuref
 * @ref:	Pointer to the reference count
 *
 * Return: The number of held references (0 ... N)
 */
static inline unsigned int rcuref_read(rcuref_t *ref)
{
	unsigned int c = atomic_read(&ref->refcnt);

	/* Return 0 if within the DEAD zone. */
	return c >= RCUREF_RELEASED ? 0 : c;
}

extern __must_check bool rcuref_get_slowpath(rcuref_t *ref, unsigned int new);

/**
 * rcuref_get - Acquire one reference on a rcuref reference count
 * @ref:	Pointer to the reference count
 *
 * Similar to atomic_inc_not_zero() but saturates at RCUREF_MAXREF.
 *
 * Provides no memory ordering, it is assumed the caller has guaranteed the
 * object memory to be stable (RCU, etc.). It does provide a control dependency
 * and thereby orders future stores. See documentation in lib/rcuref.c
 *
 * Return:
 *	False if the attempt to acquire a reference failed. This happens
 *	when the last reference has been put already
 *
 *	True if a reference was successfully acquired
 */
static inline __must_check bool rcuref_get(rcuref_t *ref)
{
	/*
	 * Unconditionally increase the reference count. The saturation and
	 * dead zones provide enough tolerance for this.
	 */
	unsigned int old = atomic_fetch_add_relaxed(1, &ref->refcnt);

	/*
	 * If the old value is less than RCUREF_MAXREF, this is a valid
	 * reference.
	 *
	 * In case the original value was RCUREF_NOREF the above
	 * unconditional increment raced with a concurrent put() operation
	 * dropping the last reference. That racing put() operation
	 * subsequently fails to mark the reference count dead because the
	 * count is now elevated again and the concurrent caller is
	 * therefore not allowed to deconstruct the object.
	 */
	if (likely(old < RCUREF_MAXREF))
		return true;

	/* Handle the cases inside the saturation and dead zones */
	return rcuref_get_slowpath(ref, old);
}

extern __must_check bool rcuref_put(rcuref_t *ref);

#endif
