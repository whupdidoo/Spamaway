
import libsvm._

import java.io._
import scala.collection.immutable.Map
import scala.collection.mutable.HashMap
import scala.collection.mutable.Set
import java.util.regex.Matcher
import java.util.regex.Pattern

object Spamaway {
	var svm_model_file_name = "svm_model.blob"
	var feature_model_file_name = "features_model.blob"
	var classes = Array("SPAM", "NOSPAM")
	var numClasses = classes.length
	var tokenizer: Pattern = Pattern.compile("(\\S+)")
	var matcher: Matcher = null
	var feature_space: Array[String] = null
	
	def main(args: Array[String]): Unit = {
		def usage {
			println("Usage: spamaway train <spamdir> <hamdir>\n or spamaway classify <dir>")
		}
		def action_by_arg(arg: String, params: Array[String]){
			def read_dir(directory: String): Array[(String,String)] = {
				var dir: File = new File(directory)
				if(!dir.isDirectory) {
					throw new IllegalArgumentException(directory + " is not a directory");
				}
				dir.listFiles.map{f => (f.getName, io.Source.fromFile(f)("iso-8859-1").mkString)}
			}
			
			arg match {
				case "train" =>
					println("training...")
					train (read_dir(params(0)), read_dir(params(1)))
				case "classify" => 
					println("classifying...")
					classify (read_dir(params(0)))
				case _ => usage
			}
		}
		if (args.length < 2)
			usage
		else
			action_by_arg(arg = args(0), params = args.slice(1,args.size))
	}
		
	def train(spam: Array[(String,String)], ham: Array[(String,String)]) {
		var token_space: Set[String] = Set()
		var spam_counts, ham_counts: HashMap[String, HashMap[String, Int]] = new HashMap[String, HashMap[String, Int]]
		
		//build feature space and lists of token counts per document per class
		spam.foreach{ doc =>
			if (!spam_counts.contains(doc._1))
				spam_counts.put(doc._1, new HashMap[String, Int])
			// add all tokens in this document to local token set "spam_counts(doc._1)"
			count_tokens(doc, spam_counts(doc._1))
			// add all tokens in this document to global token set "token_space"
			token_space ++= spam_counts(doc._1).keys.map{e: String => e.toLowerCase}
		}
		
		ham.foreach{ doc =>
			if (!ham_counts.contains(doc._1))
				ham_counts.put(doc._1, new HashMap[String, Int])
			count_tokens(doc, ham_counts(doc._1))
			token_space ++= ham_counts(doc._1).keys.map{e: String => e.toLowerCase}
		}
		
		feature_space = token_space.toArray //+ extra features
		var numTrainVectors = spam.size + ham.size
		var trainVectors : Array[Array[svm_node]] = new Array(numTrainVectors);
		var trainVectorClasses : Array[Double] = new Array(numTrainVectors);
		
		//fill training vectors
		var doc: (String, String) = null
		var currTrainVector: Set[svm_node] = null
		var node: svm_node = null
		var doc_counts: HashMap[String, Int] = null
		
		for(i <- 0 until spam.size ) {
			doc = spam(i)
			doc_counts = spam_counts(doc._1)
			currTrainVector = Set()
			trainVectorClasses(i) = classes.indexOf("SPAM")

			var j=0
			while (j < feature_space.size) {
				node = new svm_node
				node.index = j

				if (doc_counts.contains(feature_space(j))) {
					node.value = doc_counts(feature_space(j))
					currTrainVector += node
				}
				j += 1
			}
			//var offset = (i % 2) * 0.9f
			//currTrainVector(0).value += offset	// so that classes are separable, with 10% overlap
			//problem.addExample(currTrainVector, trainVectorClasses(i))
			trainVectors(i) = currTrainVector.toArray
		}
		
		for(i <- 0 until ham.size ) {
			doc = ham(i)
			doc_counts = ham_counts(doc._1)
			currTrainVector = Set()
			trainVectorClasses(i) = classes.indexOf("NOSPAM")
			
			var j = 0
			while (j < feature_space.size) {
				node = new svm_node
				node.index = j

				if (doc_counts.contains(feature_space(j))) {
					node.value = doc_counts(feature_space(j))
					currTrainVector += node
				}
				j +=1
			}
			//var offset = (i % 2) * 0.9f
			//currTrainVector(0).value += offset	// so that classes are separable, with 10% overlap
			//problem.addExample(currTrainVector, trainVectorClasses(i))
			trainVectors(i + spam.size) = currTrainVector.toArray
		}
		
		// scale each dimeansion to [0..1]
		var extremaForeEachDimension : (Array[Double], Array[Double]) = scaleTrainVectors(trainVectors, feature_space.length);
		// TODO: save extrema to file, for later scaling of test data? or just call scaleData(testVectors, extremaForeEachDimension._1, extremaForeEachDimension._2) if extremaForEachDimension are still available in RAM

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
		param.gamma = 1.0 / feature_space.size // width of rbf
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
		
		// 10-fold cross validation
		println("performing 10-fold cross validation...");
		var crossValidationPredictions = new Array[Double](numTrainVectors)	// CV-results are stored here
		svm.svm_cross_validation(prob,param,10,crossValidationPredictions)
		var total_correct = 0;
		for(i <- 0 until numTrainVectors)
				if(crossValidationPredictions(i) == prob.y(i))
					total_correct = total_correct+1;
		println("Cross Validation Accuracy = "+100.0*total_correct/numTrainVectors+"%");
		println("training on whole training set")

		var model = svm.svm_train(prob, param)
		svm.svm_save_model(svm_model_file_name, model)
		
		var fout: FileOutputStream = new FileOutputStream(feature_model_file_name)
		var out: ObjectOutputStream = new ObjectOutputStream(fout)
		out.writeObject(feature_space)
		fout.close
	}
	
	def classify(documents: Array[(String, String)]){
		def getFeatures(doc: (String,String)): Array[svm_node] = {		
			var counts: HashMap[String,Int] = new HashMap()			
			count_tokens(doc, counts)

			//create feature vector
			var x: Array[svm_node] = new Array[svm_node](feature_space.size)
			for (j <- 0 until feature_space.size) {
				x(j) = new svm_node
				x(j).index = j
				if (counts.contains(feature_space(j))) {
					x(j).value = counts(feature_space(j))				
				}
			}
			return x
		}
		
		var model = svm.svm_load_model(svm_model_file_name)

		//load feature space model
		var fin: FileInputStream = new FileInputStream(feature_model_file_name)
		var in: ObjectInputStream = new ObjectInputStream(fin)
		feature_space = in.readObject.asInstanceOf[Array[String]]
		fin.close
		
		documents.foreach { doc =>			 
			var v = svm.svm_predict(model, getFeatures(doc))
			println("%s %s".format(doc._1, classes((v.toInt+1)/2)))
		}
	}
	
	def count_tokens(doc: (String, String), counts: HashMap[String, Int]) {
		var token: String = null
		matcher = tokenizer.matcher(doc._2)			
		while(matcher.find){
			token = matcher.group(1)
			
			//var htmlre = "".r
			//token match 
			if (counts.contains(token)) counts(token)+=1
			else counts(token)=1
		}
	}

	def scaleTrainVectors(trainVectors : Array[Array[svm_node]], numDimensions : Int) : (Array[Double], Array[Double]) =
	{
		var minima : Array[Double] = Array.fill(numDimensions){java.lang.Double.MAX_VALUE}
		var maxima : Array[Double] = Array.fill(numDimensions){- java.lang.Double.MAX_VALUE}
		// find minima and maxima for each dimension
		trainVectors.foreach{trainVector : Array[svm_node] =>
			trainVector.foreach{svmnode : svm_node =>
				if(svmnode.value < minima(svmnode.index))
					minima(svmnode.index) = svmnode.value
				if(svmnode.value > maxima(svmnode.index))
					maxima(svmnode.index) = svmnode.value
			}
		}
		scaleData(trainVectors, minima, maxima)
		return (minima, maxima)
	}
	
	def scaleData(dataVectors : Array[Array[svm_node]], minima : Array[Double], maxima : Array[Double])
	{
		// apply linear scaling to [0..1] for each dimension: x_scaled = (x_nonscaled-min)/(max-min)
		dataVectors.foreach{dataVector : Array[svm_node] =>
			dataVector.foreach{svmnode : svm_node =>
			if(maxima(svmnode.index) == minima(svmnode.index))
				svmnode.value = 0.0	// this feature is useless, there is no variance. do not perform scaling to avoid division by 0
			else
				svmnode.value = (svmnode.value - minima(svmnode.index))/(maxima(svmnode.index) - minima(svmnode.index))
			}
		}
	}
}
