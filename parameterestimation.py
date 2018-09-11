"""
Copyright (C) 2018 Jeyaprakash Rajagopal <jeyaprakash dot rajagopal at smail dot h-brs dot de>

Version 1.0 

Dynamic model parameter estimation/identification

This module estimates the dynamic model parameters based on the Newton-Euler formulation technique. 

command:
	python parameterestimation.py

Remarks: This implementation is based on the article by Atkeson's 
'Estimation of inertial parameters of manipulator loads and links'
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from kinematicchain import *

def createYoubotChain():
	kinematic_chain = KinematicChain()
	jntAxis  = [1, 1, 1, 1, 1, -1]
	rotAxis = ['z', 'z', 'z', 'z', 'z', 'z']
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0., 0.]], np.float64), [0., 0., 0.]), FixedJoint()))
	kinematic_chain.addSegment(Segment(Link(np.array([[0.024, 0., 0.115]], np.float64), [0., 0., np.pi]), RevoluteJoint(rotAxis[1], jntAxis[1])))
	kinematic_chain.addSegment(Segment(Link(np.array([[0.033, 0., 0.]], np.float64), [-np.pi/2, 0., np.pi/2]), RevoluteJoint(rotAxis[2], jntAxis[2])))
	kinematic_chain.addSegment(Segment(Link(np.array([[0.155, 0., 0.]], np.float64), [-np.pi/2, 0., 0.]), RevoluteJoint(rotAxis[3], jntAxis[3])))
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0.135, 0.]], np.float64), [0., 0., 0.]), RevoluteJoint(rotAxis[4], jntAxis[4])))
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0.1136, 0.]], np.float64), [0., 0., -np.pi/2]), RevoluteJoint(rotAxis[5], jntAxis[5])))
	return kinematic_chain


def createTestRobotChain():
	kinematic_chain = KinematicChain()
	jntAxis  = 1
	rotAxis = 'z'
	
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0., 1.]], np.float64), [0., 0., np.pi/4]), FixedJoint()))
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0., 1.]], np.float64), [0., 0., 0.]), RevoluteJoint(rotAxis, jntAxis)))
	kinematic_chain.addSegment(Segment(Link(np.array([[0., 0., 0.]], np.float64), [0., 0., 0.]), RevoluteJoint(rotAxis, jntAxis)))
	
	return kinematic_chain


class ParameterEstimation:
	"""The main class to estimate/identify the dynamic model parameters
	
	This class is the class estimating the parameters based on Newton-Euler formulation
	Atrributes:
		self.kinematic_chain - represents object of the KinematicChain class
		self.JNTS - to store the number of joints in the manipulator excluding fixed arm joint
		self.jntAxis - represents the direction of rotation
	"""
	def __init__(self, kinematic_chain):
		"""The constructor for the ParameterEstimation class"""
		self.kinematic_chain = kinematic_chain
		self.noOfSegments = self.kinematic_chain.getNumberOfSegments()
		#Number of joints in the manipulator including a fixed joint
		self.JNTS = self.kinematic_chain.getNumberOfJoints()
		self.NOOFUNKNOWNS = 10
		self.FIXED_JOINT_1 = 1
		self.GROUND_BODY = 1
		self.BASE_BODY = 1
		#number of samples of joint information
		self.JI_N = 0


	def __del__(self):
		"""The destructor for the ParameterEstimation class"""
		del self.kinematic_chain
		print self.__class__.__name__, "Destroyed"
		del self


	def getPosesTwistAcc(self, q, qd, qdd):
		"""To get the segments of the manipulator

		Note: 
			Body angular velocity and acceleration are retrieved as well
		
		Args: 
			q - Joint angle
			qd - Joint velocity
			qdd - Joint acceleration
		
		Returns:
			segmentTransformation, segmentVelTwist, segmentAccTwist - Homogeneous transformation matrix between the links, 
			angular velocity and acceleration of the body
		"""
		kinematic_chain = self.kinematic_chain.getSegments()
		segmentTransformation, segmentVelTwist, segmentAccTwist = [], [], []

		for segment in range(len(kinematic_chain)):
			#To get the the transformation between the coordinate frames
			segmentTransformation.append(kinematic_chain[segment].getTransformation(q[segment]))
			#To get the velocity twist(linear-before-angular) based on the joint type and its axis			
			segmentVelTwist.append(kinematic_chain[segment].getVelocityTwist(qd[segment]))
			#To get the acceleration twist based on the joint type and its axis
			segmentAccTwist.append(kinematic_chain[segment].getAcceleration(qdd[segment]))
		
		return segmentTransformation, segmentVelTwist, segmentAccTwist


	def extractRotationAndTranslation(self, T):
		"""To split the homogeneous transformation matrix into rotation, translation 
		
		This function obtains the rotation, translation from the projection based matrix
		
		Args:
			T - The homogeneous transformation matrix that needs to be splitted
			
		Returns:
			rotation, translation - Rotation matrix and translation vector respectively
		"""
		rotation = T[:3,:3]
		translation = T[:3,3:4]

		return rotation, translation


	def skewSymmetricMatrix(self, v):
		"""To create the skew-symmetric matrix
		
		This function creates the skew-symmetric matrix from the vector
		
		Args: 
			v - Vector to form a skew-symmetric matrix
			
		Returns:
			skewSymmetric - skew-symmetric matrix
		"""

		skewSymmetric = np.array([
			[ 0    , -v[2],  v[1]],
			[ v[2] ,  0   , -v[0]],
			[-v[1] ,  v[0],  0   ]
			], np.float64)

		return skewSymmetric


	def getWrenchTransformation(self, rotation, translation):
		"""To get the wrench transmission matrix
		
		This method creates the general force transform between the links

		Args:
			rotation, translation - Rotation matrix and translation vector obtained from the kinematic chain
		Returns:
			wrenchTransformation - The mapping matrix of size 6x6
		"""

		RT = np.transpose(rotation)
		#Initialization of the final wrench transmission matrix
		wrenchTransmission = np.zeros((6, 6), np.float64)
		#c00
		wrenchTransmission[0:3, 0:3] = RT
		#c10
		wrenchTransmission[3:6, 0:3] = np.dot((-1 * RT), self.skewSymmetricMatrix(translation))
		#c11
		wrenchTransmission[3:6, 3:6] = RT
		
		#Return the transformation matrix of the specified joint
		return wrenchTransmission


	def getMotionTransformation(self, rotation, translation):
		"""To get the motion transformation matrix
		
		This method creates the general motion transform that computes the velocity, acceleration in body coordinates
		
		Args:
			rotation, translation - Rotation matrix and translation vector obtained from the kinematic chain
			
		Returns:
			motionTransmission - The mapping matrix of size 6x6
		"""
		RT = np.transpose(rotation)

		motionTransmission = np.zeros((6, 6), np.float64)
		motionTransmission[0:3, 0:3] = RT
		motionTransmission[0:3, 3:6] = np.dot((-1 * RT), self.skewSymmetricMatrix(translation))
		motionTransmission[3:6, 3:6] = RT
		
		#Return the transformation matrix of the specified joint
		return motionTransmission
	
	
	def dotOmega(self, angVelOrAcc):
		"""To compute .w matrix which is the part of the simplification

		Args: 
			angVelOrAcc - Parameter represents the angular velocity or acceleration
			
		Returns:
			array with the 3x6 matrix
		"""
		return np.array([
					[angVelOrAcc[0], angVelOrAcc[1], angVelOrAcc[2], 0, 0, 0],
					[0, angVelOrAcc[0], 0, angVelOrAcc[1], angVelOrAcc[2], 0],
					[0, 0, angVelOrAcc[0], 0, angVelOrAcc[1], angVelOrAcc[2]]
					], np.float64)


	def Acceleration(self, linearAcc, w, wd):
		"""To compute the acceleration matrix obtained through the Newton-Euler formulation
		
		This method computes the acceleration matrix that is given in the formulation

		Args: 
			linearAcc - Body's linear acceleration matrix
			w - Body's angular velocity
			wd - Body's angular acceleration

		Returns:
			acceleration - Acceleration matrix in the size 6x10
		"""	
		#Process: angular velocity matrix
		wm = self.skewSymmetricMatrix(w)
		#Process: angular acceleration matrix
		wdm = self.skewSymmetricMatrix(wd)
		
		#linear acceleration from a list to an array
		linearAcceleration = np.asarray(linearAcc).reshape(3,1)
		#Initializing the acceleration matrix 6x10 with zeroes
		acceleration= np.zeros((6, 10), np.float64)
		#c00
		acceleration[0:3, 0:1] = linearAcceleration
		#c01
		acceleration[0:3, 1:4] = np.add(wdm, (np.dot(wm, wm)))
		#c11
		acceleration[3:6, 1:4] = self.skewSymmetricMatrix(np.dot(linearAcceleration, -1))
		#c12
		acceleration[3:6, 4:10] = np.add(self.dotOmega(wd), np.dot(wm, self.dotOmega(w)))

		#Returns the acceleration matrix for the individual links
		return acceleration


	def observationMatrix(self, q, qd, qdd):
		"""To estimate the kinematic matrix.

		This method computes the kinematic matrix that will be later used in the rigde regression

		Args:
			q - Joint angle
			qd - Joint velocity
			qdd - Joint acceleration

		Returns:
			kinematicMatrix - the kinematic model matrix
		"""
		#Transformations for all the links
		rotationWrtPrevFrame, translationWrtPrevFrame = [], []
		#Lists to store motion, wrench transmission matrices
		motionTransmission, wrenchTransmission = [], []
		poseArrBwLinks, jntVelTwist, jntAccTwist = self.getPosesTwistAcc(q, qd, qdd)
		jntAccTwist.append(np.array([0, 0, 0, 0, 0, 0], np.float64))
		print "jntAccTwist\n", jntAccTwist
		
		#Initialization:Angular velocity in the body coordinates
		velBody = []
		#Initialization:Angular acceleration in the body coordinates
		accBody = []
		#Motion transmission matrix from ground to base which is an identity(forward recusion)
		motionTransmission.append(np.identity(6))
		#Wrench transmission matrix from base to ground (backward recursion)
		wrenchTransmission.append(np.identity(6))

		#Motion and wrench transmission matrices computation
		for index in range(self.noOfSegments):
			#To store the pose of the segment temporarily 
			tmpRotation, tmpTranslation = self.extractRotationAndTranslation(poseArrBwLinks[index])
			print 'I am here'
			print tmpRotation, tmpTranslation
						
			#Get the motion transmission matrix
			motionTransmission.append(self.getMotionTransformation(tmpRotation, tmpTranslation))
			#Get the wrench transmission matrix
			wrenchTransmission.append(self.getWrenchTransformation(tmpRotation, tmpTranslation))

		print poseArrBwLinks[0]
		print poseArrBwLinks[1]
		print poseArrBwLinks[2]
		print "T \n", np.dot(np.dot(poseArrBwLinks[0], poseArrBwLinks[1]), poseArrBwLinks[2])

		#To compute the forward velocity kinematics
		#List that starts with the velocity of the ground body
		velBody.append(np.array([0., 0., 0., 0., 0., 0.], np.float64))
		#Computes velocity for all the rigid bodies of the manipulator
		for index in range(1, self.noOfSegments + self.GROUND_BODY):
			#Motion transmission index represents the transmission from link 1 frame to base body frame since
			#there is an additional body attached as ground for the convention followed in this work
			velBody.append(np.add(np.dot(motionTransmission[index], velBody[index - 1]), jntVelTwist[index - 1]))
		print velBody
		#To compute the forward acceleration kinematics
		#To store the spatial cross product of the twist
		vcaptwist = np.zeros((6, 6), np.float64)
		#List starts with the acceleration of the ground body
		accBody.append(np.array([0, 0, 9.81, 0, 0, 0], np.float64))
		#Loop that computes the acceleration of the manipulator segments
		for index in range(1, self.noOfSegments + self.GROUND_BODY):
			#Translational velocity of the rigid body starting from the body base
			vx = self.skewSymmetricMatrix(velBody[index][0:3])
			#Angular velocity of the rigid body starting from the body base
			wx = self.skewSymmetricMatrix(velBody[index][3:6])
			#Spatial cross product of the twist (vx) 
			vcaptwist[0:3, 0:3] = wx
			vcaptwist[0:3, 3:6] = vx
			vcaptwist[3:6, 3:6] = wx
			#Computation for the acceleration of the rigid bodies 
			#motionTransmission starts with index instead of index-1 signifies from base frame to link 1 
			#body attached frame 
			tmpVj = np.dot(motionTransmission[index-1], jntVelTwist[index - 1])
			print "tf "
			print np.dot(motionTransmission[index], accBody[index - 1])
			tmpAccBodyTwist = np.add(np.dot(motionTransmission[index], accBody[index - 1]), jntAccTwist[index-1])
			accBody.append(np.add(tmpAccBodyTwist, np.dot(vcaptwist, tmpVj)))
			#print "accbody:: \n", accBody[index]

		#Acceleration for all the links
		A = []
		#This work includes the ground segment's acceleration for the generality of this software 
		for index in range(self.noOfSegments + self.GROUND_BODY):
			wixvi = np.dot(self.skewSymmetricMatrix(velBody[index][3:6]), velBody[index][0:3])
			result = accBody[index][0:3]#np.subtract(accBody[index][0:3], wixvi)
			A.append(self.Acceleration(result, velBody[index][3:6], accBody[index][3:6]))

		#### DEBUGGING ####
		K = np.zeros((self.JNTS, self.JNTS * self.NOOFUNKNOWNS), np.float64)
		print "wrenchTrans : \n", wrenchTransmission

		for row in range(self.JNTS):
			accumulatedTransformation = np.identity(6)
			for col in range(row, self.JNTS):
				#Transformation is identical for the diagonal elements in the observation matrix
				if (row == col):
					U = A[col + self.GROUND_BODY + self.BASE_BODY]

				#Accumulated transformation to project the joint effort from proximal to distal links
				else:
					accumulatedTransformation = np.dot(accumulatedTransformation, wrenchTransmission[col + self.GROUND_BODY + self.BASE_BODY])
					print "before accumulation : \n", accumulatedTransformation
					#Computing the elements of the observation matrix
					U = np.dot(accumulatedTransformation, A[col + self.GROUND_BODY + self.BASE_BODY])
				print "acc trans: \n", accumulatedTransformation
				#Project every element of the observation matrix to the joint's internal axis
				k = self.kinematic_chain.getSegments()[col + self.BASE_BODY].joint.projectAcceleration(U)
				#Patching the observation matrix elements together as a single matrix
				K[row:row+1, self.NOOFUNKNOWNS * col:self.NOOFUNKNOWNS * (col + 1)] = k
		
		#Returns the observation or kinematic matrix
		return K


	def printAllLinkEstimates(self, l):
		""" To display the estimation results.

		To display the estimated parameters after ridge regression.

		Args: 
			l1Res ... l5Res - represents the estimation in the links from 1 to 5

		Returns:
			Displaying the end results in the terminal.
		"""
		print ' '
		fmt = ''
		for index in range(self.JNTS + self.FIXED_JOINT_1):
			fmt += '{:<17}'
		
		#fmt = '{:<17}{:<17}{:<17}{:<17}{:<17}{:<17}'
		parameters = ['m(kg)', 'm*c_x(kg.m)', 'm*c_y(kg.m)', 'm*c_z(kg.m)', 'Ixx(kg.m^2)', 'Ixy(kg.m^2)', 'Ixz(kg.m^2)', 'Iyy(kg.m^2)', 'Iyz(kg.m^2)', 'Izz(kg.m^2)']
		
		#print(fmt.format('Parameters', 'Link1', 'Link2', 'Link3', 'Link4', 'Link5'))
		print '\033[0;0m'

		for param,l1,l2,l3,l4,l5 in zip(parameters,l[0],l[1],l[2],l[3],l[4]):
			print(fmt.format(param, l1, l2, l3, l4, l5))


	def getMotionTorqueDataFromFile(self):
		""" To read joint information such as joint position, velocity and torque
		
		This function reads the input from a file
		
		Returns: 
			jntPos, jntVel, jntTrq - Joint position, velocity, acceleration respectively
		"""
		#number of rows in the input file
		row = 0
		#for indexing purposes
		countPos, countVel, countTrq = 0, 0, 0
		self.JI_N = 1503
		#creating 2-D array for storing the number of experiments with the number of experiments conducted 
		jntPos, jntVel, jntTrq = [[] for _ in range(self.JI_N)], [[] for _ in range(self.JI_N)], [[] for _ in range(self.JI_N)]

		with open('joint_info.dat','r') as f:
			df=pd.DataFrame(l.rstrip().split() for l in f)
		
		#To iterate through the rows and split the position, velocity and acceleration inputs from the input file
		for index, row in df.iterrows():
			if row[0] == 'Position':
				for i in range(1, len(row)):
					jntPos[countPos].append(row[i]) #this gives me the joint angle for the joint 1...5
				countPos = countPos + 1
			elif row[0] == 'Velocity':
				for i in range(1, len(row)):
					jntVel[countVel].append(row[i]) #this gives me the joint velocity for the joint 1...5
				countVel = countVel + 1
			elif row[0] == 'Torque':
				for i in range(1, len(row)):
					jntTrq[countTrq].append(row[i]) #this gives me the joint torque for the joint 1...5
				countTrq = countTrq + 1

		#The number of joints are observed based on the number of inputs on the input file
		#self.JNTS = (len(row))
		#Returns the joint position, velocity and acceleration 
		return jntPos, jntVel, jntTrq


	def getTorquesFromAllJoints(self, trq):
		""" To compute the torque by projecting it into the joint coordinates
		
		This method computes the torque by projecting into the joint's internal coordinates
		
		Args: 
			trq - Individual joint torque values
			
		Returns: 
			finalTrq - the final torque colum matrix
		"""
		torques = []

		for index in range(1, self.JNTS + self.FIXED_JOINT_1):
			torques.append(trq[index])
		
		#To stack the torques for all the joints of the manipulator vertically as a single matrix for the computations
		finalTrq = np.vstack(torques)
		#Returns the nx1 torque column vector
		return finalTrq


	def offsetCalculation(self, pos):
		""" To compute the offset between the model conventions

		This function computes the offsets between the different model representations used by different authors

		Args:
			pos - Joint positions

		Returns:
			jPos - Joint positions which is the result of the offset computations
		"""
		jPos = []
		jPos.append(0.0)
		offsets = [0.0, 2.9496, 1.1345, -2.5482, 1.7890, 2.9234]
		for i in range(1, self.JNTS+1):
			jPos.append(pos[i] - offsets[i])
			
		return jPos


	def Estimation(self):
		"""The main method that estimates the dynamic model parameters
		
		This is the main function that estimates/identifies the dynamic robot model parameters
		
		Returns:
			The individual estimates of the links to print
		"""
		os.system('clear')
		print ' '

		print("\t\t   " + "\033[1mEstimation/Identification of the inertial parameters")

		jntPosFromFile, jntVelFromFile, jntTrqFromFile = self.getMotionTorqueDataFromFile()
		
		jntPos, jntVel, jntTrq, jntAcc = [[] for _ in range(self.JI_N)], [[] for _ in range(self.JI_N)], [[] for _ in range(self.JI_N)], [[] for _ in range(self.JI_N)]
		dt = 0.11
		accAtHomeposition = 0.
		kinematicMatComputation = [[] for _ in range(self.JI_N)]
	
		for j in range(self.JI_N):
			for i in range(self.JNTS + self.FIXED_JOINT_1):
				if i == 0:
					jntPos[j].append(np.float64(0.))
					jntVel[j].append(np.float64(0.))
					jntTrq[j].append(np.float64(0.))

				if i != (self.JNTS):
					jntPos[j].append(np.float64(jntPosFromFile[j][i]))
					jntVel[j].append(np.float64(jntVelFromFile[j][i]))
					jntTrq[j].append(np.float64(jntTrqFromFile[j][i]))

				if j == 0:
					jntAcc[0].append(accAtHomeposition)
				else:
					jntAcc[j].append((jntVel[j][i] - jntVel[j-1][i])/dt)

			jntPos[j] = self.offsetCalculation(jntPos[j])
			tempKinematicCompuation  = self.observationMatrix(jntPos[j], jntVel[j], jntAcc[j])
			kinematicMatComputation[j].append(tempKinematicCompuation)

		#axis = 1 because the axis along which the arrays are merged
		kinematicTmpComputation = np.concatenate(kinematicMatComputation, axis = 1)
		# deleting one dimension from the (1, rows, colums) axis0 is deleted
		K = np.squeeze(kinematicTmpComputation, axis = 0)
		KT = np.transpose(K)

		#print KT.shape
		trq = np.vstack(jntTrq)

		torqueComputation = [[] for _ in range(self.JI_N)]

		for i in range(self.JI_N):
			torqueComputation[i].append(self.getTorquesFromAllJoints(trq[i]))

		T =  np.concatenate(torqueComputation, axis = 1)

		TFinal = np.squeeze(T, axis = 0)

		#Least square estimate
		estimateFinalSub = np.dot(KT, K)
		#print estimateFinalSub
		#estimating the eigen value of the matrix K^T.K
		w = np.linalg.eigvals(estimateFinalSub)
		#print w
		#Defining the threshold to ignore the close to zero values for both positive and negative eigenvalues
		tol = 1e-3
		real_w = w.real[abs(w.real) > tol]

		#printing the smallest nonzero value of K^T*K
		eigenvalue = np.amin(real_w)

		#Minimum eigenvalue
		eigenMin = estimateFinalSub + ((eigenvalue * 0.003) * np.identity(self.JNTS * self.NOOFUNKNOWNS))
		#Computations related to the ridgeregression
		finalResult = np.dot(np.linalg.inv(eigenMin), np.dot(KT, TFinal))

		#spliting the array of estimated parameters of the individual links into separate arrays
		l = np.split(finalResult, self.JNTS)

		#print l1Res, l2Res, l3Res, l4Res, l5Res
		#print np.split(finalResult, 5)
		#to display the estimated inertial parameters
		self.printAllLinkEstimates(l)

def parallelAxisTheorem(rotInertiaExpressedInCOM, mass, comExpressedInJointFrame):
	c = np.array(comExpressedInJointFrame)
	I = np.array(rotInertiaExpressedInCOM)
	cTc = np.dot(c.transpose(), c)
	
	ccT = np.dot(c, c.transpose())
	print "ccT"

	cTcI = (cTc * np.identity(3))

	rotInertiaExpressedInJointFrame = np.add(I, np.dot(mass, np.subtract(cTcI, ccT)))
	print np.shape(rotInertiaExpressedInJointFrame)
	return rotInertiaExpressedInJointFrame

def main():
	### Parameter estimation
	
	# 2-link manipulator
	objLink = ParameterEstimation(createTestRobotChain())
	kinematicMatrix = objLink.observationMatrix([0., 1.0, 1.0], np.array([0., 0., 0.], np.float64), np.array([0., 0., 0.], np.float64))
	
	# Psi unknown model parameters
	m = 1
	c = [[1], [1], [1]]
	I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	#I = [[1], [0], [0], [1], [0], [1]]
	inertialMat = []
	
	I = parallelAxisTheorem(I, m, c)
	print np.shape(I)
	#print "I \n", I

	inertialMat = []

	for index in range(3):
		#norm = np.linalg.norm(I[index, 0:3])
		for j in range(index, 3):
			inertialMat.append(I[index, j])
	
	print "inertialMat \n", inertialMat

	print np.shape(inertialMat)
	inertialMat = np.asarray(inertialMat).reshape(6, 1)
	c = (m * c)
	mc = np.vstack((m, c))
	#print np.shape(mc)
	p = np.vstack((mc, np.array(inertialMat)))
	
	print "shape " , np.dot(kinematicMatrix[0,10:20], p)#kinematicMatrix[0,:]
	#p = np.vstack((mc, I))
	p = np.vstack((p, p))
	print "p ", np.shape(p)
	print "kinematic matrix : "
	print np.shape(kinematicMatrix)
	print kinematicMatrix.transpose()
	print " P:\n", p

	print 'Torque : ', np.dot(kinematicMatrix, p)
	
	"""
	#q = np.array([0.0, 2.949606436, 1.134464014, -2.548180708, 1.788962483, 2.879793266], np.float64)
	#q[3] += 1.570796327
	#q = np.array([0., 0., 0., 0., 0., 0.], np.float64)
	q = np.array([0.0, 5.8992, 2.7053, -5.1836, 3.5779, 5.8469], np.float64)
	q = objLink.offsetCalculation(q)
	qdot = np.array([0., 0., 0., 0., 0., 0.], np.float64)

	a, b, c = objLink.getPosesTwistAcc(q, qdot,qdot)
	print 'T' + '\n'
	t1 = np.dot(a[0], a[1])
	t2 = np.dot(t1, a[2])
	t3 = np.dot(t2, a[3])
	t4 = np.dot(t3, np.dot(a[4], a[5]))
	print t4
	"""
	"""
	# youbot manipulator
	objLink = ParameterEstimation(createYoubotChain())

	objLink.Estimation()
	"""

if __name__ == "__main__":
	main()
