import libsvm._

import java.io._
import java.util._

object Spamaway {
	var model_file_name = "svm_model.blob"
	var classes = Array("SPAM", "NOSPAM")
	var numClasses = classes.length
	var numTrainVectors = 100
	var numFeatures = 2
		
	def main(args: Array[String]): Unit = {
		def usage {
			println("Usage: spamaway train <spamdir> <hamdir>\n or spamaway classify <dir>")
		}
		case class Argument(value :String)
		def action_by_arg(arg: Argument, params: Array[String]){
			def read_dir(directory: String): Array[String] = {
				var dir: File = new File(directory)
				if(!dir.isDirectory)	{
					throw new IllegalArgumentException(directory + " is not a directory");
				}
				dir.listFiles.map{f => io.Source.fromFile(f)("iso-8859-1").toString}
			}
			
			arg.value match {
				case "train" =>
					//TODO: add scaling (!)
					train(read_dir(params(0)), read_dir(params(1)))
				case "classify" => 
					classify(read_dir(params(0)))
				case _ => usage
			}
		}
		if (args.length < 2)
			usage
		else
			action_by_arg(arg = Argument(args(0)), params = args.slice(1,args.size))
	}
	
	def train(spam: Array[String], ham: Array[String]) {
		var trainVectors : Array[Array[svm_node]] = new Array(numTrainVectors);
		var trainVectorClasses : Array[Double] = new Array(numTrainVectors);

		for(i <- 0 to numTrainVectors-1)
		{
			var currTrainVector = trainVectors(i);
			currTrainVector = new Array[svm_node](numFeatures)
			//currTrainVector.indexes = (0 until numTrainVectors-1).toArray
			// every second item is of the same class [-1, 1]
			trainVectorClasses(i) = (i % 2)*2-1
			//currTrainVector = Array.fill(numDimensions){ var ran = java.lang.Math.random; var node = new svm_node; node.index ran.floatValue }
			for (j <- 0 until numFeatures){
				var ran = java.lang.Math.random
				var node = new svm_node
				node.index = j
				node.value = ran.floatValue
				currTrainVector(j) = node
			}
			var offset = (i % 2) * 0.9f
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
		param.kernel_type = svm_parameter.LINEAR	//0 -- linear: u'*v 1 -- polynomial: (gamma*u'*v + coef0)^degree	2 -- radial basis function: exp(-gamma*|u-v|^2)	3 -- sigmoid: tanh(gamma*u'*v + coef0)
		param.degree = 3
		param.gamma = 0	// 1/num_features
		param.gamma = 1.0 / numFeatures // width of rbf
		param.coef0 = 0
		param.nu = 0.5
		param.cache_size = 100
		param.C = 1
		param.eps = 1e-3
		param.p = 0.1
		param.shrinking = 1
		param.probability = 0
		param.nr_weight = 0
		param.weight_label = new Array[Int](0)
		param.weight = new Array[Double](0)

		var error_msg = svm.svm_check_parameter(prob, param)
		if(error_msg != null) {
			println("ParamCheckError: "+error_msg+"\n")
		}

		var model = svm.svm_train(prob, param)
		svm.svm_save_model(model_file_name, model)
	}
	
	def classify(documents: Array[String]){
		def getFeatures(doc: String): Array[svm_node] = {
			//var st: StringTokenizer = new StringTokenizer(doc," \t\n\r\f");
			var x: Array[svm_node] = new Array[svm_node](numFeatures)
			for (j <- 0 until numFeatures) {
				var ran = java.lang.Math.random
				x(j) = new svm_node
				x(j).index = j
				x(j).value = ran.floatValue
			}
			return x
		}
		var model = svm.svm_load_model(model_file_name)
		for (doc <- documents){	
			var v = svm.svm_predict(model, getFeatures(doc))
			println(v)
		}
	}

}
