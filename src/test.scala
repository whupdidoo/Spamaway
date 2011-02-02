import edu.berkeley.compbio.jlibsvm.binary._
import edu.berkeley.compbio.jlibsvm.kernel._
import edu.berkeley.compbio.jlibsvm._
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

import java.util.HashSet

object test
{
	def main(args: Array[String])
	{
		var labels = Array("SPAM", "HAM")
		// SparseVector gave me no errors, so i kept it. SparseVector is a java class from jlibsvm
		var train1 = new SparseVector(2)
		var train2 = new SparseVector(2)
		train1.indexes = Array(0, 1)
		train1.values = Array(0.1f, 0.3f)
		train1.indexes = Array(0, 1)
		train1.values = Array(0.8f, 0.5f)
		
		var trainDataElement = Array(train1, train2);
		
		var problem = new MutableBinaryClassificationProblemImpl[String, SparseVector](labels.getClass, 2)
		// this is a for-loop in scala, not uncool
		(0 until 1).foreach( i => problem.addExample(trainDataElement(i), labels(i)) )
		//problem.setupLabels();	// why does this throw an IndexOutOfBoundsException?		var svm = new C_SVC[String,SparseVector]
		var builder : ImmutableSvmParameterGrid.Builder[String, SparseVector] = ImmutableSvmParameterGrid.builder()
		builder.nu = 0.5f;
		builder.cache_size = 100;
		builder.eps = 1e-3f;
		builder.p = 0.1f;
		builder.shrinking = true;
		builder.probability = false;
		builder.redistributeUnbalancedC = true;
		builder.kernelSet = new HashSet()
		builder.kernelSet.add(new LinearKernel())	// others are also possible
		var cSet = List(1.0f,2.0f);
		builder.Cset = new java.util.ArrayList()
		cSet.foreach(el => builder.Cset.add(el))	// copy scala list to java list
		var param = builder.build
		var svm = new C_SVC[String, SparseVector]	// also possible: Nu_SVC, EpsilonSVR, Nu_SVR
		var model = svm.train(problem, param)
	}
}