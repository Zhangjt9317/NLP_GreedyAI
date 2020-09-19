def get_review_summary_for_business(biz_id):
	# 获取每一个business的评论总结
	

def main(): 

       	bus_ids = []  # 指定几个business ids

	for bus_id in bus_ids:
		print ("Working on biz_id %s" % bus_id)
		start = time.time()

		summary = get_review_summary_for_business(bus_id)
		
		# format and print....

if __name__ == "__main__":
	main()


