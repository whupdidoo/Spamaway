
import libsvm._

import java.io._
import scala.collection.immutable.Map
import scala.collection.mutable.HashMap
import scala.collection.mutable.Set
import java.util.regex.Matcher
import java.util.regex.Pattern

class FeatureSpace {
	var header_list: Array[String] = null
	var token_space: Array[String] = null
	var scaling: (Array[Double], Array[Double]) = null

	def extract_token_features(body: String): Array[Double] = {
		var features : Array[Double] = new Array(token_space.size)
		// hashmap for faster access
		var token_hashmap : HashMap[String, Int] = new HashMap[String, Int]()
		for(i <- 0 until token_space.size)
			token_hashmap(token_space(i)) = i
		var token: String = null
		matcher = Pattern.compile("(\\S+)").matcher(body)
		while(matcher.find){
			token = matcher.group(1)
			var feature_index = token_hashmap(token)
			if(feature_index != null)
				features(feature_index) += 1
		}
		return features
	}

	def extract_header_features
	{

	}

	def get_feature(index: Int): String = {
		if (index >= header_list.size)
			return token_list(index)
		else
			return header_list(index)
	}
	
	def size: Int = {
		header_list.size + token_list.size
	}
}

object Spamaway {
	var svm_model_file_name = "svm_model.blob"
	var feature_model_file_name = "features_model.blob"
	var classes = Array("SPAM", "NOSPAM")
	var numClasses = classes.length
	var tokenizer: Pattern = Pattern.compile("(\\S+)")
	var matcher: Matcher = null
	var feature_space: FeatureSpace = new FeatureSpace
	var scale_factors : (Array[Double], Array[Double]) = null
	
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
					performCrossValidation = (params.size == 3 && params(3).toLowerCase().equals("cv"))
					train (read_dir(params(0)), read_dir(params(1)), performCrossValidation)
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
		
	def train(spam: Array[(String,String)], ham: Array[(String,String)], performCrossValidation : Boolean) {
		var tokens: Set[String] = Set()
		
		//build token space
		Array.concat(spam, ham).foreach{ doc =>
			matcher = tokenizer.matcher(doc._2)
			while(matcher.find){
				token = matcher.group(1)
				tokens += token
			}
		}

		var feature_space = new FeatureSpace
		feature_space.token_space = tokens.toArray

		// TODO: build header space
		var trainVectors : Array[Array[svm_node]] = new Array()
		var trainVectorClasses = Array.fill(spam.size)(0) ++ Array.fill(ham.size)(1)
		(spam ++ ham).map{doc =>
			var (header, body) = decapitate(doc)
			var featureVector : Array[Double] =	feature_space.extract_header_features(header)
				++ feature_space.extract_token_features(body)
			var trainVector : Array[svm_node] = new Array()
			var i = 0;
			while(i < featureVector.size)
			{
				if(featureVector(i) != 0.0)
				{
					var node = new svm_node
					node.index = i
					node.value = featureVector(i)
					trainVector += node
				}
			}
			trainVectors += trainVector
		}
		
		// scale each dimension to [0..1]
		scale_factors = scaleTrainVectors(trainVectors, feature_space.size)
		feature_space.scaling = scale_factors

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

		if(performCrossValidation)
		{
			println("performing 10-fold cross validation...")
			var crossValidationPredictions = new Array[Double](numTrainVectors)	// CV-results are stored here
			svm.svm_cross_validation(prob,param,10,crossValidationPredictions)
			var total_correct = 0;
			for(i <- 0 until numTrainVectors)
					if(crossValidationPredictions(i) == prob.y(i))
						total_correct = total_correct+1
			println("Cross Validation Accuracy = "+100.0*total_correct/numTrainVectors+"%")
		}
		
		println("training on whole training set")
		var model = svm.svm_train(prob, param)
		svm.svm_save_model(svm_model_file_name, model)
		
		//write extra data
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
			var x: Set[svm_node] = Set()
			for (j <- 0 until feature_space.size) {
				if (counts.contains(feature_space.get_feature(j))) {
					var n = new svm_node 
					n.value = counts(feature_space.get_feature(j))
					n.index = j
					x.add(n)
				}
			}
			return x.toArray
		}
		
		var model = svm.svm_load_model(svm_model_file_name)

		//load feature space model
		var fin: FileInputStream = new FileInputStream(feature_model_file_name)
		var in: ObjectInputStream = new ObjectInputStream(fin)
		feature_space.header_list = in.readObject.asInstanceOf[Array[String]]
		feature_space.token_list = in.readObject.asInstanceOf[Array[String]]
		scale_factors = in.readObject.asInstanceOf[(Array[Double], Array[Double])]
		fin.close
			
		documents.foreach { doc =>
			var features = getFeatures(doc)		
			scaleData(Array(features), scale_factors._1, scale_factors._2)
			var v = svm.svm_predict(model, features)
			println("%s %s".format(doc._1, classes((v.toInt+1)/2)))
		}
	}

	def decapitate(doc : String): (String, String) = {
		var matcher : Matcher = Pattern.compile("(.+)\\r?\\n\\r?\\n(.+)").matcher
		if(matcher.find){
			var header : String = matcher.group(1)
			var body : String = matcher.group(2)
			return (header, body)
		}
		throw new IllegalArgumentException("no email header / body found")
		return null
	}

	/*
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
	*/

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
		var numUselessFeatures = 0
		for(i <- 0 until numDimensions)
		{
			if(minima(i) == maxima(i))
				numUselessFeatures += 1
		}
		println("percentage of useless features: " + 100.0*numUselessFeatures/numDimensions+"%")
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
