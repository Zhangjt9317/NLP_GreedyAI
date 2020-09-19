


class Business(object):
	"""
	用来表示跟business相关的变量和函数
	"""

	SENTIMENT_MODEL = SentimentModel() # 把已经训练好的模型存放在文件里，并导入进来
	

	def __init__(self, review_df):
		# 初始化变量以及函数


	def aspect_based_summary(self):
		"""
		返回一个business的summary. 针对于每一个aspect计算出它的正面负面情感以及TOP reviews. 
		具体细节请看给定的文档。 
		"""

		

		return {'business_id': 
				'business_name': 
				'business_rating': 
				'aspect_summary': 	
				}


	def extract_aspects(self):
		"""
		从一个business的review中抽取aspects
		"""



