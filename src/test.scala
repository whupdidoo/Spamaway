import edu.berkeley.compbio.jlibsvm.binary._
import edu.berkeley.compbio.jlibsvm.kernel._
import edu.berkeley.compbio.jlibsvm._
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint.Builder
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

import scala.runtime.RichFloat

import java.util.HashSet

object test
{
	def main(args: Array[String])
	{
		var classes = Array("SPAM", "HAM")
		// SparseVector gave me no errors, so i kept it. SparseVector is a java class from jlibsvm
		var numClasses = classes.length
		var numTrainVectors = 100
		var numDimensions = 2
		var trainVectors : Array[SparseVector] = new Array(numTrainVectors);
		var trainVectorClasses : Array[String] = new Array(numTrainVectors);
		var problem = new MutableBinaryClassificationProblemImpl[String, SparseVector](classes.getClass, numClasses)
		for(i <- 0 to numTrainVectors-1)
		{
			var currTrainVector = trainVectors(i);
			currTrainVector = new SparseVector(numDimensions)
			currTrainVector.indexes = (0 until numTrainVectors-1).toArray
			// every second item is of the same class
			trainVectorClasses(i) = classes(i % 2);
			currTrainVector.values = Array.fill(numDimensions){ var ran = java.lang.Math.random; ran.floatValue }
			var offset = (i % 2) * 0.9f;
			currTrainVector.values(0) += offset	// so that classes are separable, with 10% overlap
			problem.addExample(currTrainVector, trainVectorClasses(i))
		}
		
		//problem.setupLabels();
		var builder : ImmutableSvmParameterPoint.Builder[String, SparseVector] = new ImmutableSvmParameterPoint.Builder
		builder.nu = 0.5f;
		builder.cache_size = 100;
		builder.eps = 1e-3f;
		builder.p = 0.1f;
		builder.shrinking = true;
		builder.probability = false;
		builder.redistributeUnbalancedC = true;
		//builder.kernelSet = new HashSet()
		//builder.kernelSet.add(new LinearKernel())	// others are also possible
		builder.kernel = new LinearKernel()
		builder.C = 1.0f;
		//var cSet = List(1.0f);
		//builder.Cset = new java.util.ArrayList()
		//cSet.foreach(el => builder.Cset.add(el))	// copy scala list to java list
		var param = builder.build
		var svm = new C_SVC[String, SparseVector]	// also possible: Nu_SVC, EpsilonSVR, Nu_SVR
		svm.validateParam(param)
		//var model = svm.train(problem, param)
		var model = svm.trainOne(problem, 1.0f, 1.0f, param)
	}
}