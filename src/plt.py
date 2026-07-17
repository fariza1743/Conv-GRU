	plt.figure(figsize=(10, 5))
	plt.plot(labels, label='Ground Truth', linewidth=2)
	plt.plot(predictions, label='Predictions', linewidth=2, alpha=0.7)
	plt.title(f'Patient {last_patient} – Run {last_run}')
	plt.xlabel('Sample Index')
	plt.ylabel('Glucose Level (mg/dL)')
	plt.legend()
	plt.tight_layout()

	# Show and save
	plt.show()
	plt.savefig(f'pred_vs_label_patient_{last_patient}_run_{last_run}.png', dpi=300)
	print(f"Saved plot: pred_vs_label_patient_{last_patient}_run_{last_run}.png")