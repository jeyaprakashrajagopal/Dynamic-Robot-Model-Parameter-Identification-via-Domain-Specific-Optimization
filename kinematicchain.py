"""
Copyright (C) 2018 Jeyaprakash Rajagopal <jeyaprakash dot rajagopal at smail dot h-brs dot de>

Version 1.0 

Kinematic chain creation for the robot manipulator

This module forms the kinematic chain for the robot manipulator with the given kinematic information
and it can be used for obtaining the transformations between the segments based on the user provided 
joint angle information. This implementation is for both fixed and revolute joints which can be 
extended with the prismatic joint.

command:
	python kinematicchain.py

ToDo's:
	The kinematic chain of the youBot manipulator can be extended further for the base also.
"""
import numpy as np

def getTaitBryanAngle(theta):
	"""To convert from the extrinsic(fixed-frame convention) to intrinsic(moving frame convention).

	Note: 
		The euler angles are used in youBot-store model but this work uses three distinct angles
		for rotation which is based 
	Args: 
		theta - relative orientation of the body frame
	Returns:
		Homogeneous transformation matrix for a fixed joint.
	"""
	# Cos and sin declaration
	ca, sa = np.cos(theta[2]), np.sin(theta[2])
	cb, sb = np.cos(theta[1]), np.sin(theta[1])
	cc, sc = np.cos(theta[0]), np.sin(theta[0])
	# rotation matrix
	R = np.array([[cb * cc, -cb * sc, sb], 
				 [sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb], 
				 [-ca * sb * cc + sa * sc,  ca * sb * sc + sa * cc,  ca * cb]], np.float64)

	return R


class FixedJoint:
	"""Transformation and twist computation for the fixed joint.
	
	This module implements the functionality of a fixed joint.
	"""
	def getTransformation(self, q):
		"""The transformation for the fixed joint.
		
		Note: 
			The rotation matrix is an identity for this kind of a joint.
		Args: 
			q - Joint angle
		Returns:
			The homogeneous transformation matrix.
		"""
		return np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
	
	def getVelocityTwist(self, qd):
		"""Get the angular velocity twist of a fixed joint.
		
		Note: 
			The angular velocity twist is zero since the joint is fixed.
		Args: 
			qd - Joint space velocity
		Returns:
			Cartesian space angular velocity matrix.
		"""
		return np.transpose(np.array([0, 0, 0, 0, 0, 0], np.float64))
	
	def getAcceleration(self, qdd):
		"""Get the angular acceleration of a fixed joint.
		
		Note: 
			The angular acceleration is zero since the joint is fixed.
		Args: 
			qdd - Joint space acceleration
		Returns:
			Cartesian space angular acceleration matrix.
		"""
		return np.transpose(np.array([0, 0, 0, 0, 0, 0], np.float64))


class RevoluteJoint:
	"""Transformation and twist computation for the revolute joint.
	This class reflects the functionality of a revolute joint. 
	
	Note: 
		The rotation matrix is transposed since the reference frame is referred 
		from the i coordinate system to i+1 coordinate system
		
	Attributes:
		axis - represents the joint's axis of rotation.
	"""
	def __init__(self, axis, direction):
		self.rotaxis = axis
		self.direction = direction

		if self.rotaxis == 'z':
			self.subSpaceMatrix = np.array([0, 0, 0, 0, 0, 1], np.float64)
		elif self.rotaxis == 'y':
			self.subSpaceMatrix = np.array([0, 0, 0, 0, 1, 0], np.float64)
		elif self.rotaxis == 'x':
			self.subSpaceMatrix = np.array([0, 0, 0, 1, 0, 0], np.float64)

		self.subSpaceMatrixToJointAxis = self.subSpaceMatrix * self.direction

	def getTransformation(self, q):
		"""The transformation for the revolute joint based on it's axis of rotation.

		Note: 
			The rotation matrix is transposed.
		Args: 
			q - Joint angle
		Returns:
			The homogeneous transformation matrix.
		"""
		ca, sa = np.cos(q), np.sin(q)
		T = []
		if self.rotaxis  == 'z' and self.direction == 1:
			T = np.array([[ca, -sa, 0, 0],[sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
		elif  self.rotaxis  == 'z' and self.direction == -1:
			T = np.array([[ca, sa, 0, 0],[-sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
		elif self.rotaxis == 'y' and self.direction == 1:
			T = np.array([[ca, 0, sa, 0],[0, 1, 0, 0], [-sa, 0, ca, 0], [0, 0, 0, 1]], np.float64)
		elif self.rotaxis == 'y' and self.direction == -1:
			T = np.array([[ca, 0, -sa, 0],[0, 1, 0, 0], [sa, 0, ca, 0], [0, 0, 0, 1]], np.float64)
		elif self.rotaxis == 'x' and self.direction == 1:
			T = np.array([[1, 0, 0, 0],[0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]], np.float64)
		elif self.rotaxis == 'x' and self.direction == -1:
			T = np.array([[1, 0, 0, 0],[0, ca, sa, 0], [0, -sa, ca, 0], [0, 0, 0, 1]], np.float64)
		return T

	def getVelocityTwist(self, qd):
		"""Get the angular velocity twist of the revolute joint.
		
		Note: 
			The angular velocity twist is zero since the joint is fixed.
		Args: 
			qd - Joint space velocity
		Returns:
			(self.subSpaceMatrix * qd) - Cartesian space angular velocity matrix.
		"""
		return self.subSpaceMatrixToJointAxis * qd

	def getAcceleration(self, qdd):
		"""Get the angular acceleration of a revolute joint joint.
		
		Note: 
			The angular acceleration is zero since the joint is fixed.
		Args: 
			qdd - Joint space acceleration
		Returns:
			(self.subSpaceMatrix * qdd) - Cartesian space angular acceleration matrix.
		"""	
		return self.subSpaceMatrixToJointAxis * qdd

	def projectWrench(self, wrench):
		"""To project the wrench to torque since there is no force sensor in the manipulator.
		
		Args: 
			6x1 wrench coordinate vector
		Returns:
			torque - The projected torque value is returned.
		"""
		torque = np.dot(self.subSpaceMatrixToJointAxis.reshape(1,6), wrench)
		return torque
	
	def projectAcceleration(self, spatialAcceleration):
		"""The inverse acceleration kinematics that returns the joint torque.
		
		Args: 
			6x10 acceleration matrix
		Returns:
			jointAcceleration - The joint acceleration of the joint.
		"""

		jointAcceleration = np.dot(self.subSpaceMatrix.reshape(1,6), spatialAcceleration)
		return jointAcceleration


class KinematicChain:
	"""To add and get the segments for creating the kinematic chain.
	
	This class gets the segments and forms the kinematic chain and then returns the 
	complete kinematic chain of the manipulator.
	
	Attributes:
		segments - This stores the relative references of each segment
	"""
	def __init__(self):
		self.segments = []
	
	def addSegment(self, segment):
		"""To add the segments to the kinematic chain of the robot manipulator.
		Args: 
			segments - The particular segment that needs to be added in the chain.
		Returns:
			None.
		"""
		self.segments.append(segment)
	
	def getSegments(self):
		"""To return collection of segments that has had been formed as a kinematic chain.
		
		Note: 
			The references to the instances are formed as a chain in a programming 
			perspective.
		Args: 
			None
		Returns:
			self.segments - The kinematic chain that has been created.
		"""
		return self.segments

	def getNumberOfJoints(self):
		"""To return the number of joints in the kinematic chain.
		
		Args: 
			None
		Returns:
			joints - The number of joints in the manipulator.
		"""		
		joints = 0
		
		for segment in self.getSegments():
			if not isinstance(segment.joint, FixedJoint):
				joints += 1
		
		return joints

	def getNumberOfSegments(self):
		"""To return the number of segments in the kinematic chain.
		
		Args: 
			None
		Returns:
			segments- The number of joints in the manipulator.
		"""
		assert (len(self.segments) > 0), "Please create the segment(s) first."
		return len(self.segments)

class Segment:
	"""The segments are being created with the help of the link, joint transformations.
	
	This class creates the segments and helps creating the kinematic chain and the velocity,
	acceleration twist vectors in the body coordinates.
	
	Attributes:
		self.link - reference to the link class.
		self.joint - reference to the joint class.
	"""	
	def __init__(self, link, joint):
		self.link = link
		self.joint = joint
		
	def getTransformation(self, q):
		"""The transformation for the revolute joint based on it's axis of rotation.
		
		Note: 
			The rotation matrix is transposed.
		Args: 
			q - Joint angle
		Returns:
			The homogeneous transformation matrix.
		"""		
		return np.dot(self.joint.getTransformation(q), self.link.getTransformation())
	
	def getVelocityTwist(self, qd):
		"""Get the angular velocity twist of the revolute joint.
		
		Note: 
			The angular velocity twist is zero since the joint is fixed.
		Args: 
			qd - Joint space velocity
		Returns:
			Cartesian space angular velocity matrix.
		"""
		return self.joint.getVelocityTwist(qd)
		
	def getAcceleration(self, qdd):
		"""Get the angular acceleration of a revolute joint joint.
		
		Note: 
			The angular acceleration is zero since the joint is fixed.
		Args: 
			qdd - Joint space acceleration
		Returns:
			Cartesian space angular acceleration matrix.
		"""					
		return self.joint.getAcceleration(qdd)


class Link:
	"""To compute the transformations of the link based on it's kinematic specifications.
	This class reflects the functionality of a revolute joint. 
	
	Note: 
		The rotation matrix is transposed since the reference frame is referred 
		from the i coordinate system to i+1 coordinate system.
		
	Attributes:
		self.translation - represents the relative translation vector.
		self.rotation - describes the relative orientation between the references.
	"""
	def __init__(self, translation, rotation):
		self.translation = translation
		self.rotation = rotation

	def getTransformation(self):
		"""The transformation of the link based on the kinematic specifications.
		
		Note: 
			Extrinsic to instrinsic conversion as per the conventions used in this project.

		Returns:
			The link's homogeneous transformation matrix.
		"""	
		g = getTaitBryanAngle(self.rotation)
		T = np.hstack((g, np.transpose(self.translation)))
		a = np.array([0, 0, 0, 1], np.float64)
		return np.vstack((T, a))
