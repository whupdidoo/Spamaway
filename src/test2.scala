import libsvm._

import java.io._
import java.util._

object Test2 {
  def main(args: Array[String]): Unit = {
		var classes = Array("SPAM", "NOSPAM")
		var numClasses = classes.length
		var numTrainVectors = 100
		var numDimensions = 2
		var trainVectors : Array[Array[svm_node]] = new Array(numTrainVectors);
		var trainVectorClasses : Array[Double] = new Array(numTrainVectors);
		//var problem = new MutableBinaryClassificationProblemImpl[String, SparseVector](classes.getClass, numClasses)

		for(i <- 0 to numTrainVectors-1)
		{
			var currTrainVector = trainVectors(i);
			currTrainVector = new Array[svm_node](numDimensions)
			//currTrainVector.indexes = (0 until numTrainVectors-1).toArray
			// every second item is of the same class
			trainVectorClasses(i) = i % 2;
			//currTrainVector = Array.fill(numDimensions){ var ran = java.lang.Math.random; var node = new svm_node; node.index ran.floatValue }
			for (j <- 0 until numDimensions){
				var ran = java.lang.Math.random
				var node = new svm_node
				node.index = j
				node.value = ran.floatValue
				currTrainVector(j) = node
			}
			var offset = (i % 2) * 0.9f;
			currTrainVector(0).value += offset	// so that classes are separable, with 10% overlap
			//problem.addExample(currTrainVector, trainVectorClasses(i))
                        trainVectors(i) = currTrainVector
		}

		var prob = new svm_problem()
		prob.l = numTrainVectors
		prob.x = trainVectors
		prob.y = trainVectorClasses


		//problem.setupLabels();
/*		var builder : ImmutableSvmParameterPoint.Builder[String, SparseVector] = new ImmutableSvmParameterPoint.Builder
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
		var param = builder.build*/
		var param : svm_parameter = new svm_parameter
		// default values
		param.svm_type = svm_parameter.C_SVC	// needs no params as i see it...
		param.kernel_type = svm_parameter.LINEAR;	//0 -- linear: u'*v 1 -- polynomial: (gamma*u'*v + coef0)^degree	2 -- radial basis function: exp(-gamma*|u-v|^2)	3 -- sigmoid: tanh(gamma*u'*v + coef0)
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.gamma = 1.0 / numDimensions;	// width of rbf
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new Array[Int](0)
		param.weight = new Array[Double](0)
		param.kernel_type = svm_parameter.LINEAR

//		var error_msg = svm.svm_check_parameter(prob,param);
//		println(error_msg)
//		if(error_msg.equals(null))
//		{
//			println("ParamCheckError: "+error_msg+"\n");
//		}

		var model = svm.svm_train(prob,param);
		//svm.svm_save_model(model_file_name,model);
	}

}
