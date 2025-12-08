"""
PMFBY Yield Prediction Engine
Report Generation Module

Generates comprehensive PDF reports for insurance officers
with yield predictions, stress analysis, and claim recommendations.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for ReportLab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, ListFlowable, ListItem
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF generation disabled.")


class PMFBYReportGenerator:
    """
    Generates professional PDF reports for PMFBY yield prediction.
    
    Sections:
    1. Title & Executive Summary
    2. Farm/Village Details
    3. Vegetation Analysis (with charts)
    4. Weather Analysis
    5. Stress Assessment
    6. Yield Prediction
    7. PMFBY Loss Calculation
    8. Recommendation
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if REPORTLAB_AVAILABLE:
            self._init_styles()
    
    def _init_styles(self):
        """Initialize PDF styles."""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='TitleCustom',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a5276')
        ))
        
        self.styles.add(ParagraphStyle(
            name='HeadingCustom',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyCustom',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#e74c3c'),
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='Success',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#27ae60'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(
        self,
        claim_data: Dict,
        yield_prediction: Dict,
        vegetation_analysis: Dict,
        weather_analysis: Dict,
        stress_analysis: Dict,
        pmfby_loss: Dict,
        aggregation: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            claim_data: Farm/claim details
            yield_prediction: Prediction results
            vegetation_analysis: NDVI/LAI analysis
            weather_analysis: Weather stress analysis
            stress_analysis: Combined stress indices
            pmfby_loss: PMFBY loss calculation
            aggregation: Optional aggregation data
            
        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Generating text report.")
            return self._generate_text_report(
                claim_data, yield_prediction, vegetation_analysis,
                weather_analysis, stress_analysis, pmfby_loss
            )
        
        # Generate filename
        claim_id = claim_data.get('claim_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"PMFBY_Report_{claim_id}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        
        # Title Page
        story.extend(self._build_title_page(claim_data))
        
        # Executive Summary
        story.extend(self._build_executive_summary(
            yield_prediction, pmfby_loss, stress_analysis
        ))
        
        # Farm Details
        story.extend(self._build_farm_details(claim_data))
        
        # Vegetation Analysis
        story.extend(self._build_vegetation_section(vegetation_analysis))
        
        # Weather Analysis
        story.extend(self._build_weather_section(weather_analysis))
        
        # Stress Assessment
        story.extend(self._build_stress_section(stress_analysis))
        
        # Yield Prediction
        story.extend(self._build_yield_section(yield_prediction))
        
        # PMFBY Loss Calculation
        story.extend(self._build_pmfby_section(pmfby_loss))
        
        # Recommendation
        story.extend(self._build_recommendation(
            pmfby_loss, yield_prediction, stress_analysis
        ))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated report: {filepath}")
        
        return filepath
    
    def _build_title_page(self, claim_data: Dict) -> List:
        """Build title page elements."""
        elements = []
        
        elements.append(Spacer(1, 50))
        elements.append(Paragraph(
            "PMFBY Yield Prediction Report",
            self.styles['TitleCustom']
        ))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(
            "AI-Based Crop Yield Assessment System",
            self.styles['Heading2']
        ))
        elements.append(Spacer(1, 40))
        
        # Claim details table
        claim_info = [
            ['Claim ID:', claim_data.get('claim_id', 'N/A')],
            ['Farmer Name:', claim_data.get('farmer_name', 'N/A')],
            ['Crop Type:', claim_data.get('crop_type', 'N/A').upper()],
            ['Season:', claim_data.get('season', 'N/A')],
            ['District:', claim_data.get('district', 'N/A')],
            ['Village:', claim_data.get('village', 'N/A')],
            ['Sowing Date:', claim_data.get('sowing_date', 'N/A')],
            ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M')]
        ]
        
        table = Table(claim_info, colWidths=[3*cm, 8*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(PageBreak())
        
        return elements
    
    def _build_executive_summary(
        self,
        yield_pred: Dict,
        pmfby_loss: Dict,
        stress: Dict
    ) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['HeadingCustom']))
        
        # Key metrics table
        yield_val = yield_pred.get('yield_pred', 0)
        loss_pct = pmfby_loss.get('loss_percentage', 0)
        claim_trigger = pmfby_loss.get('claim_trigger', False)
        confidence = yield_pred.get('confidence_score', 0)
        
        metrics = [
            ['Predicted Yield:', f"{yield_val} kg/ha"],
            ['Threshold Yield:', f"{pmfby_loss.get('threshold_yield', 0)} kg/ha"],
            ['Loss Percentage:', f"{loss_pct:.1f}%"],
            ['Claim Trigger:', '✓ YES' if claim_trigger else '✗ NO'],
            ['Confidence Score:', f"{confidence:.1%}"]
        ]
        
        table = Table(metrics, colWidths=[4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9f9')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # Recommendation highlight
        if claim_trigger:
            rec_text = f"<b>RECOMMENDATION: APPROVE CLAIM</b><br/>Loss of {loss_pct:.1f}% exceeds the 33% threshold."
            elements.append(Paragraph(rec_text, self.styles['Highlight']))
        else:
            rec_text = f"<b>RECOMMENDATION: NO CLAIM</b><br/>Loss of {loss_pct:.1f}% is below the 33% threshold."
            elements.append(Paragraph(rec_text, self.styles['Success']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_farm_details(self, claim_data: Dict) -> List:
        """Build farm details section."""
        elements = []
        
        elements.append(Paragraph("Farm Details", self.styles['HeadingCustom']))
        
        details = [
            ['Area Insured:', f"{claim_data.get('area_ha', 'N/A')} hectares"],
            ['Sum Insured:', f"₹{claim_data.get('sum_insured', 0):,.2f}"],
            ['GPS Coordinates:', claim_data.get('coordinates', 'N/A')],
            ['Land Type:', claim_data.get('land_type', 'Irrigated')],
        ]
        
        table = Table(details, colWidths=[4*cm, 8*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_vegetation_section(self, veg_data: Dict) -> List:
        """Build vegetation analysis section."""
        elements = []
        
        elements.append(Paragraph("Vegetation Analysis", self.styles['HeadingCustom']))
        
        # NDVI metrics
        ndvi_metrics = [
            ['Peak NDVI:', f"{veg_data.get('ndvi_peak', 0):.3f}"],
            ['Mean NDVI:', f"{veg_data.get('ndvi_mean', 0):.3f}"],
            ['NDVI AUC:', f"{veg_data.get('ndvi_auc', 0):.2f}"],
            ['Season Length:', f"{veg_data.get('season_length', 0)} days"],
            ['Peak Date:', veg_data.get('peak_date', 'N/A')],
        ]
        
        table = Table(ndvi_metrics, colWidths=[4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 10))
        
        # Crop vigor assessment
        vigor = veg_data.get('vigor_score', 50)
        if vigor >= 80:
            vigor_text = "Excellent crop vigor observed throughout the season."
        elif vigor >= 60:
            vigor_text = "Good crop vigor with some minor stress periods."
        elif vigor >= 40:
            vigor_text = "Moderate crop vigor with notable stress impacts."
        else:
            vigor_text = "Poor crop vigor indicating significant stress or damage."
        
        elements.append(Paragraph(f"<b>Crop Vigor Score:</b> {vigor}/100", self.styles['BodyCustom']))
        elements.append(Paragraph(vigor_text, self.styles['BodyCustom']))
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_weather_section(self, weather_data: Dict) -> List:
        """Build weather analysis section."""
        elements = []
        
        elements.append(Paragraph("Weather Analysis", self.styles['HeadingCustom']))
        
        weather_metrics = [
            ['Total Rainfall:', f"{weather_data.get('rain_total', 0):.1f} mm"],
            ['Mean Temperature:', f"{weather_data.get('temp_mean', 0):.1f}°C"],
            ['Heat Stress Days:', f"{weather_data.get('heat_days', 0)}"],
            ['Dry Spell (max):', f"{weather_data.get('max_dry_spell', 0)} days"],
            ['Growing Degree Days:', f"{weather_data.get('gdd_total', 0):.0f}"],
        ]
        
        table = Table(weather_metrics, colWidths=[4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_stress_section(self, stress_data: Dict) -> List:
        """Build stress assessment section."""
        elements = []
        
        elements.append(Paragraph("Stress Assessment", self.styles['HeadingCustom']))
        
        # Stage-wise stress table
        if 'stagewise' in stress_data:
            stages = stress_data['stagewise']
            table_data = [['Growth Stage', 'Stress Index', 'Days Affected']]
            
            for stage, data in stages.items():
                if stage == 'overall':
                    continue
                table_data.append([
                    stage.replace('_', ' ').title(),
                    f"{data.get('weighted_stress', 0):.2f}",
                    f"{data.get('high_stress_days', 0)}"
                ])
            
            table = Table(table_data, colWidths=[5*cm, 3*cm, 3*cm])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 10))
        
        # Stress explanation
        explanation = stress_data.get('explanation', 'No significant stress detected.')
        elements.append(Paragraph(f"<b>Analysis:</b> {explanation}", self.styles['BodyCustom']))
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_yield_section(self, yield_data: Dict) -> List:
        """Build yield prediction section."""
        elements = []
        
        elements.append(Paragraph("Yield Prediction", self.styles['HeadingCustom']))
        
        yield_metrics = [
            ['Predicted Yield:', f"{yield_data.get('yield_pred', 0):.2f} kg/ha"],
            ['Lower Bound (10%):', f"{yield_data.get('yield_low_10', 0):.2f} kg/ha"],
            ['Upper Bound (90%):', f"{yield_data.get('yield_high_90', 0):.2f} kg/ha"],
            ['Confidence:', f"{yield_data.get('confidence_score', 0):.1%}"],
            ['Model Type:', yield_data.get('model_type', 'Unknown')],
        ]
        
        table = Table(yield_metrics, colWidths=[4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eaf2f8')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3498db')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_pmfby_section(self, pmfby_data: Dict) -> List:
        """Build PMFBY loss calculation section."""
        elements = []
        
        elements.append(Paragraph("PMFBY Loss Calculation", self.styles['HeadingCustom']))
        
        loss_metrics = [
            ['Threshold Yield (TY):', f"{pmfby_data.get('threshold_yield', 0):.2f} kg/ha"],
            ['Predicted Yield:', f"{pmfby_data.get('predicted_yield', 0):.2f} kg/ha"],
            ['Shortfall:', f"{pmfby_data.get('shortfall_kg_ha', 0):.2f} kg/ha"],
            ['Loss Percentage:', f"{pmfby_data.get('loss_percentage', 0):.2f}%"],
            ['Trigger Threshold:', f"{pmfby_data.get('trigger_threshold', 33):.0f}%"],
        ]
        
        table = Table(loss_metrics, colWidths=[4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_recommendation(
        self,
        pmfby_data: Dict,
        yield_data: Dict,
        stress_data: Dict
    ) -> List:
        """Build final recommendation section."""
        elements = []
        
        elements.append(Paragraph("Recommendation", self.styles['HeadingCustom']))
        
        claim_trigger = pmfby_data.get('claim_trigger', False)
        loss_pct = pmfby_data.get('loss_percentage', 0)
        confidence = yield_data.get('confidence_score', 0.7)
        
        if claim_trigger:
            rec_style = self.styles['Highlight']
            rec_title = "✓ CLAIM APPROVED"
            rec_reason = f"Yield loss of {loss_pct:.1f}% exceeds the 33% trigger threshold."
        else:
            rec_style = self.styles['Success']
            rec_title = "✗ NO CLAIM PAYABLE"
            rec_reason = f"Yield loss of {loss_pct:.1f}% is below the 33% threshold."
        
        elements.append(Paragraph(rec_title, rec_style))
        elements.append(Paragraph(rec_reason, self.styles['BodyCustom']))
        elements.append(Spacer(1, 10))
        
        # Caveats
        elements.append(Paragraph("<b>Notes:</b>", self.styles['BodyCustom']))
        
        notes = []
        if confidence < 0.7:
            notes.append("• Low confidence prediction - recommend CCE verification")
        if loss_pct > 25 and loss_pct < 35:
            notes.append("• Loss close to threshold - recommend field inspection")
        
        notes.append("• Prediction based on satellite + weather data analysis")
        notes.append("• Final decision subject to CCE ground truth verification")
        
        for note in notes:
            elements.append(Paragraph(note, self.styles['BodyCustom']))
        
        elements.append(Spacer(1, 30))
        
        # Signature section
        elements.append(Paragraph("_" * 50, self.styles['BodyCustom']))
        elements.append(Paragraph("Authorized Signature", self.styles['BodyCustom']))
        
        return elements
    
    def _generate_text_report(
        self,
        claim_data: Dict,
        yield_pred: Dict,
        veg: Dict,
        weather: Dict,
        stress: Dict,
        pmfby: Dict
    ) -> str:
        """Generate text-only report when PDF not available."""
        claim_id = claim_data.get('claim_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"PMFBY_Report_{claim_id}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        lines = [
            "=" * 60,
            "PMFBY YIELD PREDICTION REPORT",
            "=" * 60,
            "",
            f"Claim ID: {claim_data.get('claim_id', 'N/A')}",
            f"Farmer: {claim_data.get('farmer_name', 'N/A')}",
            f"Crop: {claim_data.get('crop_type', 'N/A')}",
            f"District: {claim_data.get('district', 'N/A')}",
            "",
            "-" * 40,
            "YIELD PREDICTION",
            "-" * 40,
            f"Predicted Yield: {yield_pred.get('yield_pred', 0)} kg/ha",
            f"Range: {yield_pred.get('yield_low_10', 0)} - {yield_pred.get('yield_high_90', 0)} kg/ha",
            f"Confidence: {yield_pred.get('confidence_score', 0):.1%}",
            "",
            "-" * 40,
            "PMFBY ANALYSIS",
            "-" * 40,
            f"Threshold Yield: {pmfby.get('threshold_yield', 0)} kg/ha",
            f"Loss: {pmfby.get('loss_percentage', 0):.1f}%",
            f"Claim Trigger: {'YES' if pmfby.get('claim_trigger') else 'NO'}",
            "",
            "-" * 40,
            "RECOMMENDATION",
            "-" * 40,
            "APPROVE" if pmfby.get('claim_trigger') else "REJECT",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60
        ]
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated text report: {filepath}")
        return filepath


def main():
    """Test report generation."""
    generator = PMFBYReportGenerator("output/reports")
    
    # Sample data
    claim_data = {
        'claim_id': 'PMFBY2024001',
        'farmer_name': 'Ramesh Kumar',
        'crop_type': 'rice',
        'season': 'Kharif 2024',
        'district': 'Karnal',
        'village': 'Taraori',
        'sowing_date': '2024-06-15',
        'area_ha': 2.5,
        'sum_insured': 125000,
        'coordinates': '29.69°N, 76.97°E'
    }
    
    yield_prediction = {
        'yield_pred': 2450,
        'yield_low_10': 2100,
        'yield_high_90': 2800,
        'confidence_score': 0.78,
        'model_type': 'transformer'
    }
    
    vegetation_analysis = {
        'ndvi_peak': 0.72,
        'ndvi_mean': 0.55,
        'ndvi_auc': 11.5,
        'season_length': 125,
        'peak_date': '2024-08-20',
        'vigor_score': 65
    }
    
    weather_analysis = {
        'rain_total': 850,
        'temp_mean': 28.5,
        'heat_days': 12,
        'max_dry_spell': 8,
        'gdd_total': 1750
    }
    
    stress_analysis = {
        'stagewise': {
            'vegetative': {'weighted_stress': 0.15, 'high_stress_days': 2},
            'tillering': {'weighted_stress': 0.20, 'high_stress_days': 3},
            'flowering': {'weighted_stress': 0.45, 'high_stress_days': 8},
            'grain_filling': {'weighted_stress': 0.30, 'high_stress_days': 5},
            'maturity': {'weighted_stress': 0.10, 'high_stress_days': 1},
        },
        'explanation': 'Moderate moisture stress during flowering stage reduced yield potential.'
    }
    
    pmfby_loss = {
        'threshold_yield': 3000,
        'predicted_yield': 2450,
        'shortfall_kg_ha': 550,
        'loss_percentage': 18.3,
        'claim_trigger': False,
        'trigger_threshold': 33
    }
    
    # Generate report
    report_path = generator.generate_report(
        claim_data, yield_prediction, vegetation_analysis,
        weather_analysis, stress_analysis, pmfby_loss
    )
    
    print(f"\nReport generated: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
